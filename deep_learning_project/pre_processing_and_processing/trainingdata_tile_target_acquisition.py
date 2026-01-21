import shutil
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox

from pathlib import Path
from shapely.geometry import box
from pyproj import CRS, Transformer


# -----------------------------
# Helpers
# -----------------------------
from shapely.geometry import box

def _grid_tiles_over_polygon(projected_polygon, tile_size_m: float, min_overlap: float = 0.0):
    """
    Create square tiles over polygon bounding box, then filter.

    Filtering:
      - Always requires centroid within polygon.
      - Optionally requires overlap ratio >= min_overlap (0.0 disables overlap filter).
    """
    minx, miny, maxx, maxy = projected_polygon.bounds
    xs = np.arange(minx, maxx, tile_size_m)
    ys = np.arange(miny, maxy, tile_size_m)

    tiles = []
    for x in xs:
        for y in ys:
            t = box(x, y, x + tile_size_m, y + tile_size_m)

            # 1) centroid must be inside polygon
            if not projected_polygon.contains(t.centroid):
                continue

            # 2) optional overlap ratio threshold
            if min_overlap > 0.0:
                inter_area = t.intersection(projected_polygon).area
                if inter_area / t.area < min_overlap:
                    continue

            tiles.append(t)

    return tiles



def _largest_connected_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    if G.is_directed():
        G = ox.convert.to_undirected(G)
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    lcc_nodes = max(comps, key=len)
    return G.subgraph(lcc_nodes).copy()


def _tile_bounds_to_wsen(tile_proj, to_wgs84):
    """
    Convert projected tile bounds to EPSG:4326 bbox in OSMnx's expected order:
    (west, south, east, north) == (left, bottom, right, top)
    """
    minx, miny, maxx, maxy = tile_proj.bounds

    # Projected -> lon/lat
    west, south = to_wgs84.transform(minx, miny)
    east, north = to_wgs84.transform(maxx, maxy)

    return (west, south, east, north)


def _download_tile_graph(
    bbox,
    network_type="drive",
    simplify=True,
    retain_all=True,
    truncate_by_edge=True,
    custom_filter=None,
):
    """
    OSMnx-compatible downloader for:
    osmnx.graph.graph_from_bbox(bbox, *, network_type='all', simplify=True, retain_all=False, truncate_by_edge=False, custom_filter=None)

    bbox must be (west, south, east, north) in EPSG:4326.
    """
    # Use the graph submodule explicitly, per your signature
    G = ox.graph.graph_from_bbox(
        bbox,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        custom_filter=custom_filter,
    )
    return ox.convert.to_undirected(G)



# -----------------------------
# Pool builder: allow >20 OK per city to enable redistribution
# -----------------------------
def build_pool(
    places,
    out_dir="tile_graphs_europe_pool",
    tile_size_m=2000.0,
    min_nodes=200,
    network_type="drive",
    seed=42,
    simplify=True,
    retain_all=True,
    truncate_by_edge=True,
    max_ok_per_city_in_pool=40,   # <-- collect up to this many OK tiles per city
):
    """
    Builds a pool of OK tiles across all cities.
    Collects up to max_ok_per_city_in_pool per city (fixed tile size, no fallback).
    Saves every OK tile graph into out_dir and records metadata for all attempts.
    """
    ox.settings.use_cache = True
    ox.settings.log_console = True

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    rows = []
    global_tile_id = 0

    for place in places:
        print(f"\n=== City: {place} ===")

        gdf = ox.geocode_to_gdf(place)
        if gdf.empty:
            print(f"  - Could not geocode. Skipping: {place}")
            continue

        gdf_proj = ox.projection.project_gdf(gdf)
        poly_proj = gdf_proj.iloc[0].geometry

        proj_crs = CRS.from_user_input(gdf_proj.crs)
        to_wgs84 = Transformer.from_crs(proj_crs, CRS.from_epsg(4326), always_xy=True)

        candidates = _grid_tiles_over_polygon(poly_proj, tile_size_m=tile_size_m, min_overlap=0.5)

        if len(candidates) == 0:
            print(f"  - No candidate tiles at {tile_size_m:.0f}m. Skipping: {place}")
            continue

        cand_idx = np.arange(len(candidates))
        rng.shuffle(cand_idx)

        city_ok = 0
        attempts = 0

        for idx in cand_idx:
            if city_ok >= max_ok_per_city_in_pool:
                break

            tile = candidates[idx]
            bbox = _tile_bounds_to_wsen(tile, to_wgs84)
            bbox = _tile_bounds_to_wsen(tile, to_wgs84)
            west, south, east, north = bbox  # <-- define these for metadata

            attempts += 1

            try:
                G = _download_tile_graph(
                    bbox,
                    network_type=network_type,
                    simplify=simplify,
                    retain_all=retain_all,
                    truncate_by_edge=truncate_by_edge,
                )

                G = _largest_connected_component(G)

                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()

                if n_nodes < min_nodes or n_edges == 0:
                    rows.append({
                        "global_tile_id": None,
                        "place": place,
                        "tile_size_m": tile_size_m,
                        "north": north, "south": south, "east": east, "west": west,
                        "status": "filtered_small",
                        "error": None,
                        "n_nodes": n_nodes,
                        "n_edges": n_edges,
                        "graphml_path": None,
                    })
                    continue

                safe_city = place.split(",")[0].strip().replace(" ", "_")
                graphml_name = f"tile_{global_tile_id:05d}_{safe_city}.graphml"
                graphml_path = out_path / graphml_name
                ox.save_graphml(G, filepath=str(graphml_path))

                rows.append({
                    "global_tile_id": global_tile_id,
                    "place": place,
                    "tile_size_m": tile_size_m,
                    "north": north, "south": south, "east": east, "west": west,
                    "status": "ok",
                    "error": None,
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "graphml_path": str(graphml_path),
                })

                global_tile_id += 1
                city_ok += 1

            except Exception as e:
                rows.append({
                    "global_tile_id": None,
                    "place": place,
                    "tile_size_m": tile_size_m,
                    "north": north, "south": south, "east": east, "west": west,
                    "status": "download_failed",
                    "error": str(e),
                    "n_nodes": np.nan,
                    "n_edges": np.nan,
                    "graphml_path": None,
                })

        print(f"  -> OK tiles in pool: {city_ok}/{max_ok_per_city_in_pool} (attempts: {attempts})")
        city_rows = pd.DataFrame([r for r in rows if r["place"] == place])
        print(city_rows["status"].value_counts(dropna=False))

        # If there are download_failed entries, show a few error messages:
        errs = city_rows[city_rows["status"] == "download_failed"]["error"].dropna().head(3).tolist()
        if errs:
            print("Sample errors:", errs)


    meta = pd.DataFrame(rows)
    meta_csv = out_path / "tile_metadata_pool.csv"
    meta.to_csv(meta_csv, index=False)
    print(f"\nSaved pool metadata to: {meta_csv}")

    return meta


# -----------------------------
# Balanced selection:
# - take target_per_city when available, otherwise take what's available
# - redistribute remaining slots from leftover tiles in other cities
# -----------------------------
def balanced_select_tiles(
    pool_meta: pd.DataFrame,
    places,
    select_n=200,
    target_per_city=20,
    seed=42,
):
    rng = np.random.default_rng(seed)

    ok = pool_meta[pool_meta["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No OK tiles in the pool.")

    # Ensure we only consider requested places and keep consistent order
    ok = ok[ok["place"].isin(places)]

    # First pass: take up to target_per_city per city
    selected_parts = []
    leftover_parts = []

    for place in places:
        city_ok = ok[ok["place"] == place].copy()
        if city_ok.empty:
            continue

        idx = np.arange(len(city_ok))
        rng.shuffle(idx)

        take = min(target_per_city, len(city_ok))
        sel = city_ok.iloc[idx[:take]].copy()
        rem = city_ok.iloc[idx[take:]].copy()

        selected_parts.append(sel)
        if not rem.empty:
            leftover_parts.append(rem)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else ok.iloc[0:0].copy()

    if len(selected) > select_n:
        # If somehow first-pass exceeds select_n (e.g., too many cities), downsample globally.
        idx = rng.choice(len(selected), size=select_n, replace=False)
        return selected.iloc[idx].reset_index(drop=True)

    remaining = select_n - len(selected)
    if remaining == 0:
        return selected.reset_index(drop=True)

    # Second pass: redistribute from leftovers
    leftovers = pd.concat(leftover_parts, ignore_index=True) if leftover_parts else ok.iloc[0:0].copy()

    if len(leftovers) < remaining:
        raise RuntimeError(
            f"Not enough OK tiles to reach {select_n}. "
            f"Selected {len(selected)} in first pass, but only {len(leftovers)} leftovers available."
        )

    idx2 = rng.choice(len(leftovers), size=remaining, replace=False)
    selected2 = leftovers.iloc[idx2].copy()

    final = pd.concat([selected, selected2], ignore_index=True)
    return final.reset_index(drop=True)


def copy_selected_graphs(selected_df: pd.DataFrame, selected_dir: str):
    sel_path = Path(selected_dir)
    sel_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in selected_df.iterrows():
        src = Path(row["graphml_path"])
        if src.exists():
            shutil.copy2(src, sel_path / src.name)
            copied += 1
    return copied


# -----------------------------
# Run it
# -----------------------------
if __name__ == "__main__":
    ox.settings.use_cache = True
    ox.settings.log_console = True

    cities = [
        "Barcelona, Spain",
        "Paris, France",
        "Amsterdam, Netherlands",
        "London, United Kingdom",
        "Berlin, Germany",
        "Copenhagen, Denmark",
        "Rome, Italy",
        "Stockholm, Sweden",
        "Vienna, Austria",
        "Prague, Czechia",
    ]

    # 1) Build a pool with room for redistribution (collect up to 40 OK tiles per city)
    pool = build_pool(
        places=cities,
        out_dir="tile_graphs_europe_pool",
        tile_size_m=2000.0,
        min_nodes=200,
        network_type="drive",
        seed=42,
        max_ok_per_city_in_pool=40,   # <-- enables redistribution
    )

    # 2) Balanced select: aim for 20/city; if a city has fewer, redistribute
    selected_200 = balanced_select_tiles(
        pool_meta=pool,
        places=cities,
        select_n=200,
        target_per_city=20,
        seed=42,
    )

    # 3) Save selected metadata
    out_path = Path("tile_graphs_europe_pool")
    selected_csv = out_path / "tile_metadata_selected_200_balanced.csv"
    selected_200.to_csv(selected_csv, index=False)
    print(f"\nSaved balanced selection metadata to: {selected_csv}")

    # 4) Copy selected graphs into a clean folder
    copied = copy_selected_graphs(selected_200, selected_dir="tile_graphs_europe_selected_200_balanced")
    print(f"Copied {copied} selected GraphML files to: tile_graphs_europe_selected_200_balanced")

    # Quick diagnostics
    print("\nSelected counts per city:")
    print(selected_200["place"].value_counts())
