import argparse
import json
import os
from typing import Dict, List, Any, Tuple, Optional
import hashlib
import math
import random
import time

import numpy as np
from numpy.typing import NDArray
import pandas as pd
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def _cache_paths(json_path: str) -> Tuple[str, str, str]:
    base = os.path.splitext(json_path)[0] + "_embedding_cache"
    return base + ".npz", base + "_ann.faiss", base + "_ann_ids.npy"


def _ann_meta_path_from_cache_path(cache_path: str) -> str:
    base, _ = os.path.splitext(cache_path)
    return base + "_ann_meta.json"


def _matrix_digest(mat: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(mat.astype(np.float32)).tobytes())
    return h.hexdigest()


def load_catalog(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure in {json_path}. Expected a list or an object with key 'items' being a list.")
def get_title(item: Dict[str, Any]) -> str:
    return str(item.get("Title") or item.get("title") or "")

def get_genres_list(item: Dict[str, Any]) -> List[str]:
    genres_val = item.get("Genres")
    genres_list: List[str] = []
    if isinstance(genres_val, list):
        for g in genres_val:
            if isinstance(g, dict) and "Name" in g and g["Name"] is not None:
                genres_list.append(str(g["Name"]))
            elif g is not None:
                genres_list.append(str(g))
    elif isinstance(genres_val, str):
        genres_list = [genres_val] if genres_val else []
    elif isinstance(genres_val, dict) and "Name" in genres_val:
        if genres_val.get("Name"):
            genres_list = [str(genres_val.get("Name"))]
    return genres_list

def ann_topk_ids(
    embeddings: Dict[str, NDArray[np.float32]],
    ann_index: "faiss.Index",
    ann_ids: List[str],
    seed_id: str,
    k: int,
) -> List[str]:
    seed_vec = embeddings[seed_id].astype(np.float32).reshape(1, -1)
    D, I = ann_index.search(seed_vec, min(k + 1, len(ann_ids)))
    idxs = [i for i in I[0].tolist() if i >= 0]
    rec_ids: List[str] = []
    for idx in idxs:
        iid = ann_ids[idx]
        if iid != seed_id:
            rec_ids.append(iid)
        if len(rec_ids) >= k:
            break
    return rec_ids

def ann_topk_with_scores(
    embeddings: Dict[str, NDArray[np.float32]],
    ann_index: "faiss.Index",
    ann_ids: List[str],
    seed_id: str,
    k: int,
) -> List[Tuple[str, float]]:
    seed_vec = embeddings[seed_id].astype(np.float32).reshape(1, -1)
    D, I = ann_index.search(seed_vec, min(k + 1, len(ann_ids)))
    results: List[Tuple[str, float]] = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        iid = ann_ids[idx]
        if iid == seed_id:
            continue
        results.append((iid, float(score)))
        if len(results) >= k:
            break
    return results
def ann_query_vector_with_scores(
    query_vec: NDArray[np.float32],
    ann_index: "faiss.Index",
    ann_ids: List[str],
    k: int,
    exclude: Optional[set] = None,
    overfetch: int = 0,
) -> List[Tuple[str, float]]:
    q = query_vec.astype(np.float32).reshape(1, -1)
    fetch_n = min(max(k + (overfetch if overfetch > 0 else 0), k), len(ann_ids))
    D, I = ann_index.search(q, fetch_n)
    results: List[Tuple[str, float]] = []
    excluded = exclude or set()
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        iid = ann_ids[idx]
        if iid in excluded:
            continue
        results.append((iid, float(score)))
        if len(results) >= k:
            break
    return results


def save_embeddings(path: str, embeddings: Dict[str, NDArray[np.float32]]) -> None:
    ids = list(embeddings.keys())
    mat = np.stack([embeddings[i] for i in ids], axis=0)
    np.savez_compressed(path, ids=np.array(ids), embeddings=mat)



def _genres_to_str(genres_val: Any) -> str:
    if isinstance(genres_val, list):
        names: List[str] = []
        for g in genres_val:
            if isinstance(g, dict) and "Name" in g and g["Name"] is not None:
                names.append(str(g["Name"]))
            elif g is not None:
                names.append(str(g))
        return ", ".join(names)
    if isinstance(genres_val, str):
        return genres_val
    if isinstance(genres_val, dict) and "Name" in genres_val:
        return str(genres_val.get("Name") or "")
    return ""


def generate_embeddings(
    catalog: List[Dict[str, Any]],
    json_path: str,
    model_name: str = "sentence-transformers/all-MPNet-base-v2",
    genre_weight: float = 0.6,
    description_weight: float = 0.4,
) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, Dict[str, Any]]]:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed. Please install it before running.")

    if genre_weight < 0 or description_weight < 0 or (genre_weight + description_weight) <= 0:
        raise ValueError("genre_weight and description_weight must be >=0 and their sum > 0.")

    cache_path, _, _ = _cache_paths(json_path)

    embedding_dim: Optional[int] = None
    id_to_item: Dict[str, Dict[str, Any]] = {}

    ids: List[str] = []
    genre_texts: List[str] = []
    description_texts: List[str] = []
    item_hashes: List[str] = []

    for it in catalog:
        item_id = it.get("Id") or it.get("id") or it.get("ID")
        if item_id is None:
            continue
        item_id = str(item_id)

        title = str(it.get("Title") or it.get("title") or "").strip()
        genres_str = _genres_to_str(it.get("Genres"))
        description = str(it.get("Description") or it.get("description") or title).strip()

        if not genres_str and not description:
            continue

        id_to_item[item_id] = it
        ids.append(item_id)
        genre_texts.append(genres_str if genres_str else "")
        description_texts.append(description if description else "")

        h = hashlib.sha1()
        h.update(model_name.encode("utf-8"))
        h.update(
            b"|gw=" + repr(genre_weight).encode("utf-8")
            + b"|dw=" + repr(description_weight).encode("utf-8")
        )
        h.update(b"|genres:" + genres_str.lower().strip().encode("utf-8"))
        h.update(b"|description:" + description.lower().strip().encode("utf-8"))
        item_hashes.append(h.hexdigest())

    if not ids:
        return {}, {}

    cached_ids_list: Optional[List[str]] = None
    cached_emb: Optional[np.ndarray] = None
    cached_h: Optional[np.ndarray] = None
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False, mmap_mode="r") as npz:
                cached_ids_list = npz["ids"].astype(str).tolist()
                cached_emb = np.asarray(npz["embeddings"], dtype=np.float32)
                cached_h = npz["hashes"].astype(str)
                if cached_emb is not None:
                    embedding_dim = int(cached_emb.shape[1])
        except Exception:
            cached_ids_list = None
            cached_emb = None
            cached_h = None

    to_compute_indices: List[int] = []
    reused_vectors: Dict[int, NDArray[np.float32]] = {}
    if cached_ids_list is not None and cached_emb is not None and cached_h is not None:
        id_to_pos = {str(cid): i for i, cid in enumerate(cached_ids_list)}
        for i, item_id in enumerate(ids):
            pos = id_to_pos.get(item_id)
            if pos is not None and cached_h[pos] == item_hashes[i]:
                reused_vectors[i] = cached_emb[pos]
            else:
                to_compute_indices.append(i)
    else:
        to_compute_indices = list(range(len(ids)))

    model = None
    if to_compute_indices:
        print(f"Computing embeddings for {len(to_compute_indices)} items...")
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_sentence_embedding_dimension() if embedding_dim is None else embedding_dim

    new_vectors: Dict[int, NDArray[np.float32]] = {}
    if to_compute_indices:
        subset_genres = [genre_texts[i] for i in to_compute_indices]
        subset_descs  = [description_texts[i] for i in to_compute_indices]

        if model is None:
            raise RuntimeError("Internal error: model not initialized despite pending items to encode.")
        g_emb = model.encode(subset_genres, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        d_emb = model.encode(subset_descs,  show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)

        for j, idx in enumerate(to_compute_indices):
            g_vec = g_emb[j]
            d_vec = d_emb[j]
            
            g_norm_val = np.linalg.norm(g_vec)
            g_norm = g_vec / g_norm_val if g_norm_val > 0 else g_vec

            d_norm_val = np.linalg.norm(d_vec)
            d_norm = d_vec / d_norm_val if d_norm_val > 0 else d_vec

            combined = (genre_weight * g_norm) + (description_weight * d_norm)
            
            combined_norm_val = np.linalg.norm(combined)
            if combined_norm_val > 0:
                combined = combined / combined_norm_val
            
            new_vectors[idx] = combined.astype(np.float32)

    embeddings: Dict[str, NDArray[np.float32]] = {}
    final_matrix: List[NDArray[np.float32]] = []
    for i, item_id in enumerate(ids):
        vec = reused_vectors.get(i) if i in reused_vectors else new_vectors.get(i)
        if vec is None:
            dim = int(embedding_dim) if embedding_dim is not None else 384
            vec = np.zeros((dim,), dtype=np.float32)
        embeddings[item_id] = vec
        final_matrix.append(vec)

    if to_compute_indices:
        try:
            np.savez_compressed(
                cache_path,
                ids=np.array(ids),
                embeddings=np.stack(final_matrix, axis=0),
                hashes=np.array(item_hashes),
            )
        except Exception:
            pass

    return embeddings, id_to_item


def _build_or_load_ann(cache_path: str, embeddings: Dict[str, NDArray[np.float32]]) -> Tuple[Any, List[str]]:
    ids = sorted(embeddings.keys())
    if not ids:
        return None, []
    base, _ = os.path.splitext(cache_path)
    ann_path = base + "_ann.faiss"
    ids_path = base + "_ann_ids.npy"
    meta_path = _ann_meta_path_from_cache_path(cache_path)
    mat = np.stack([embeddings[i] for i in ids], axis=0).astype(np.float32)
    if faiss is None:
        raise ImportError("FAISS is required but not installed. Please install 'faiss-cpu'.")
    d = mat.shape[1]
    digest = _matrix_digest(mat)
    if os.path.exists(ann_path) and os.path.exists(ids_path) and os.path.exists(meta_path):
        try:
            saved_ids = np.load(ids_path, allow_pickle=False).astype(str).tolist()
            if saved_ids == ids:
                index = faiss.read_index(ann_path)
                if getattr(index, "d", None) == d and getattr(index, "ntotal", None) == len(ids):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("digest") == digest:
                        return index, saved_ids
        except Exception:
            pass
    index = faiss.IndexFlatIP(d)
    index.add(mat)
    try:
        faiss.write_index(index, ann_path)
        np.save(ids_path, np.array(ids))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"digest": digest, "d": d, "count": len(ids)}, f)
    except Exception as e:
        raise RuntimeError(f"Failed to persist FAISS index: {e}")
    return index, ids




def get_embedding_based_recommendations_ann(
    seed_id: str,
    id_to_item: Dict[str, Dict[str, Any]],
    embeddings: Dict[str, NDArray[np.float32]],
    k: int,
    cache_path: str,
) -> List[Dict[str, Any]]:
    if seed_id not in embeddings:
        raise ValueError(f"seed_id '{seed_id}' not found in embeddings.")
    ann_index, ann_ids = _build_or_load_ann(cache_path, embeddings)
    if ann_index is None or len(ann_ids) != len(embeddings):
        raise RuntimeError("FAISS index is unavailable or inconsistent. Ensure 'faiss-cpu' is installed and rebuild the index.")
    
    pairs = ann_topk_with_scores(embeddings, ann_index, ann_ids, seed_id, k)
    results: List[Dict[str, Any]] = []
    for iid, score in pairs:
        item_data = id_to_item.get(iid, {})
        title = get_title(item_data)
        genres_list = get_genres_list(item_data)
        results.append({"Id": iid, "Title": title, "Genres": genres_list, "Score": float(score)})
    return results
def build_user_profile_vector_from_interactions(
    user_id: str,
    interactions_path: str,
    embeddings: Dict[str, NDArray[np.float32]],
) -> Tuple[Optional[NDArray[np.float32]], set]:
    col_names = [
        "profileid", "contentid", "contenttype", "timestamp",
        "episodecount", "wathcedcount", "applicationid", "deviceid",
        "lang", "country",
    ]
    df = pd.read_csv(interactions_path, header=None, names=col_names, usecols=["profileid", "contentid"])
    df = df.dropna()
    df["profileid"] = df["profileid"].astype(str)
    df["contentid"] = df["contentid"].astype(str)

    user_df = df[df["profileid"] == str(user_id)]
    seen_ids = set(user_df["contentid"].tolist())
    if user_df.empty:
        return None, set()

    vecs: List[NDArray[np.float32]] = []
    for cid in seen_ids:
        v = embeddings.get(cid)
        if v is not None:
            vecs.append(v.astype(np.float32))

    if not vecs:
        return None, seen_ids

    mat = np.stack(vecs, axis=0)
    prof = np.mean(mat, axis=0)
    norm = np.linalg.norm(prof)
    if norm > 0:
        prof = prof / norm
    return prof.astype(np.float32), seen_ids

def get_user_based_recommendations_ann(
    user_id: str,
    interactions_path: str,
    id_to_item: Dict[str, Dict[str, Any]],
    embeddings: Dict[str, NDArray[np.float32]],
    k: int,
    cache_path: str,
) -> List[Dict[str, Any]]:
    ann_index, ann_ids = _build_or_load_ann(cache_path, embeddings)
    if ann_index is None or len(ann_ids) != len(embeddings):
        raise RuntimeError("FAISS index is unavailable or inconsistent. Ensure 'faiss-cpu' is installed and rebuild the index.")

    user_vec, seen_ids = build_user_profile_vector_from_interactions(
        user_id=user_id,
        interactions_path=interactions_path,
        embeddings=embeddings,
    )
    if user_vec is None:
        raise ValueError(f"Cannot build user profile: no valid interactions found in catalog for user '{user_id}'.")

    overfetch = min(len(seen_ids), max(k, 50))
    pairs = ann_query_vector_with_scores(user_vec, ann_index, ann_ids, k=k, exclude=seen_ids, overfetch=overfetch)

    results: List[Dict[str, Any]] = []
    for iid, score in pairs:
        item_data = id_to_item.get(iid, {})
        title = get_title(item_data)
        genres_list = get_genres_list(item_data)
        results.append({"Id": iid, "Title": title, "Genres": genres_list, "Score": float(score)})
    return results



def evaluate_recommender(
    embeddings: Dict[str, NDArray[np.float32]],
    id_to_item: Dict[str, Dict[str, Any]],
    interactions_path: str,
    k: int = 10,
    cache_path: Optional[str] = None,
) -> None:
    col_names = [
        "profileid", "contentid", "contenttype", "timestamp",
        "episodecount", "wathcedcount", "applicationid", "deviceid",
        "lang", "country",
    ]
    if cache_path is None:
        cache_path, _, _ = _cache_paths(interactions_path)
    ann_index, ann_ids = _build_or_load_ann(cache_path, embeddings)

    df = pd.read_csv(interactions_path, header=None, names=col_names)
    df = df[["profileid", "contentid"]].dropna()
    df["profileid"] = df["profileid"].astype(str)
    df["contentid"] = df["contentid"].astype(str)

    user_to_items: Dict[str, set] = {}
    for uid, cid in zip(df["profileid"].values, df["contentid"].values):
        s = user_to_items.get(uid)
        if s is None:
            s = set()
            user_to_items[uid] = s
        s.add(cid)

    eligible = {u: items for u, items in user_to_items.items() if len(items) >= 5}
    if not eligible:
        print("No eligible users (>=5 interactions) found for evaluation.")
        return

    def dcg(rec_ids: List[str], relevant: set, k: int) -> float:
        score = 0.0
        for i, rid in enumerate(rec_ids[:k], start=1):
            if rid in relevant:
                score += 1.0 / math.log2(i + 1)
        return score

    def idcg(num_relevant: int, k: int) -> float:
        max_hits = min(k, num_relevant)
        return sum(1.0 / math.log2(i + 1) for i in range(1, max_hits + 1))

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []

    for user, items in eligible.items():
        seed_id = random.choice(list(items))
        relevant = items - {seed_id}

        if seed_id not in embeddings:
            continue
        relevant_in_catalog = relevant.intersection(embeddings.keys())
        if not relevant_in_catalog:
            continue

        rec_ids = ann_topk_ids(embeddings, ann_index, ann_ids, seed_id, k)

        hits = sum(1 for rid in rec_ids if rid in relevant_in_catalog)
        prec = hits / float(k) if k > 0 else 0.0
        rec = hits / float(len(relevant_in_catalog)) if len(relevant_in_catalog) > 0 else 0.0

        dcg_val = dcg(rec_ids, relevant_in_catalog, k)
        idcg_val = idcg(len(relevant_in_catalog), k)
        ndcg = (dcg_val / idcg_val) if idcg_val > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg)

    if not precisions:
        print("No users with seed items present in catalog; evaluation skipped.")
        return

    avg_p = float(np.mean(precisions))
    avg_r = float(np.mean(recalls))
    avg_n = float(np.mean(ndcgs))

    print("Evaluation summary (averaged across users):")
    print(f"- Users evaluated: {len(precisions)}")
    print(f"- Precision@{k}: {avg_p:.4f}")
    print(f"- Recall@{k}:    {avg_r:.4f}")
    print(f"- NDCG@{k}:      {avg_n:.4f}")

def evaluate_user_recommender_temporal(
    embeddings: Dict[str, NDArray[np.float32]],
    id_to_item: Dict[str, Dict[str, Any]],
    interactions_path: str,
    k: int = 10,
    cache_path: Optional[str] = None,
) -> None:
    col_names = [
        "profileid", "contentid", "contenttype", "timestamp",
        "episodecount", "wathcedcount", "applicationid", "deviceid",
        "lang", "country",
    ]
    usecols = ["profileid", "contentid", "timestamp"]
    df = pd.read_csv(interactions_path, header=None, names=col_names, usecols=usecols).dropna()
    df["profileid"] = df["profileid"].astype(str)
    df["contentid"] = df["contentid"].astype(str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if cache_path is None:
        cache_path, _, _ = _cache_paths(interactions_path)
    ann_index, ann_ids = _build_or_load_ann(cache_path, embeddings)

    def dcg_single(rec_ids: List[str], target: str, k: int) -> float:
        try:
            pos = rec_ids.index(target)
            if pos < k:
                return 1.0 / math.log2((pos + 1) + 1)
        except ValueError:
            pass
        return 0.0

    precisions: List[float] = []
    recalls: List[float] = []
    ndcgs: List[float] = []
    users_evaluated = 0

    for uid, g in df.groupby("profileid"):
        g_sorted = g.sort_values("timestamp", ascending=True)
        if len(g_sorted) < 2:
            continue

        train = g_sorted.iloc[:-1]["contentid"].tolist()
        test_item = g_sorted.iloc[-1]["contentid"]

        vecs = [embeddings[cid].astype(np.float32) for cid in train if cid in embeddings]
        if not vecs:
            continue
        test_in_catalog = test_item in embeddings

        mat = np.stack(vecs, axis=0)
        prof = np.mean(mat, axis=0)
        norm = np.linalg.norm(prof)
        if norm > 0:
            prof = prof / norm
        prof = prof.astype(np.float32)

        exclude = set(train)
        overfetch = min(len(exclude), max(k, 50))
        pairs = ann_query_vector_with_scores(prof, ann_index, ann_ids, k=k, exclude=exclude, overfetch=overfetch)
        rec_ids = [iid for iid, _ in pairs]

        if test_in_catalog:
            hit = 1 if test_item in rec_ids[:k] else 0
            prec = hit / float(k) if k > 0 else 0.0
            rec = float(hit)
            dcg_val = dcg_single(rec_ids, test_item, k)
            ndcg = dcg_val
            precisions.append(prec)
            recalls.append(rec)
            ndcgs.append(ndcg)
            users_evaluated += 1

    if users_evaluated == 0:
        print("No eligible users for temporal evaluation (need >=2 interactions with in-catalog items).")
        return

    avg_p = float(np.mean(precisions)) if precisions else 0.0
    avg_r = float(np.mean(recalls)) if recalls else 0.0
    avg_n = float(np.mean(ndcgs)) if ndcgs else 0.0

    print("Temporal Evaluation summary (averaged across users):")
    print(f"- Users evaluated: {users_evaluated}")
    print(f"- Precision@{k}: {avg_p:.4f}")
    print(f"- Recall@{k}:    {avg_r:.4f}")
    print(f"- NDCG@{k}:      {avg_n:.4f}")


def pick_default_seed(embeddings: Dict[str, NDArray[np.float32]]) -> str:
    for iid in sorted(embeddings.keys()):
        return iid
    raise ValueError("No items available to pick a default seed.")


def main():
    parser = argparse.ArgumentParser(description="Embedding-based content recommender using Description + Genres.")
    parser.add_argument("--data", required=True, help="Path to JSON file (e.g., 'all 1.json').")
    parser.add_argument("--seed", required=False, help="Seed item Id. If not provided, picks the first available.")
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument("--save", required=False, help="Optional path to save computed embeddings as .npz.")
    parser.add_argument("--interactions", required=False, help="Path to interactions CSV (no header). When provided, run evaluation mode or build user profiles.")
    parser.add_argument("--user-id", required=False, help="User ID for personalized recommendations (requires --interactions).")
    parser.add_argument("--evaluate-temporal", action="store_true", help="Use temporal split evaluation instead of random seed eval.")
    parser.add_argument("--model-name", required=False, default="sentence-transformers/all-MPNet-base-v2", help="SentenceTransformer model to use.")
    parser.add_argument("--genre-weight", type=float, default=0.6, help="Weight for genre embeddings.")
    parser.add_argument("--desc-weight", type=float, default=0.4, help="Weight for description embeddings.")
    args = parser.parse_args()

    catalog = load_catalog(args.data)
    embeddings, id_to_item = generate_embeddings(
        catalog,
        json_path=args.data,
        model_name=args.model_name,
        genre_weight=args.genre_weight,
        description_weight=args.desc_weight,
    )

    if not embeddings:
        raise RuntimeError("No embeddings were generated. Check that items have Title and/or Genres.")

    if args.interactions and args.evaluate_temporal:
        cache_path, _, _ = _cache_paths(args.data)
        evaluate_user_recommender_temporal(
            embeddings=embeddings,
            id_to_item=id_to_item,
            interactions_path=args.interactions,
            k=args.k,
            cache_path=cache_path,
        )
        if args.save:
            save_embeddings(args.save, embeddings)
            print(f"Saved embeddings to: {args.save}")
        return

    if args.interactions and not args.user_id:
        cache_path, _, _ = _cache_paths(args.data)
        evaluate_recommender(
            embeddings=embeddings,
            id_to_item=id_to_item,
            interactions_path=args.interactions,
            k=args.k,
            cache_path=cache_path,
        )
        if args.save:
            save_embeddings(args.save, embeddings)
            print(f"Saved embeddings to: {args.save}")
        return

    if args.user_id:
        if not args.interactions:
            raise ValueError("--user-id requires --interactions to build the user profile.")
        cache_path, _, _ = _cache_paths(args.data)
        recs = get_user_based_recommendations_ann(
            user_id=args.user_id,
            interactions_path=args.interactions,
            id_to_item=id_to_item,
            embeddings=embeddings,
            k=args.k,
            cache_path=cache_path,
        )
        print(f"User-based recommendations for User: {args.user_id}")
        print("Top recommendations:")
        for r in recs:
            genres_val = r.get("Genres", [])
            genres_str = ", ".join(genres_val) if isinstance(genres_val, list) else str(genres_val)
            print(f"- Id: {r['Id']:<10} | Title: {r['Title']:<40} | Genres: {genres_str:<30} | Score: {r['Score']:.5f}")
        if args.save:
            save_embeddings(args.save, embeddings)
            print(f"Saved embeddings to: {args.save}")
        return

    seed_id = args.seed or pick_default_seed(embeddings)

    cache_path, _, _ = _cache_paths(args.data)
    recs = get_embedding_based_recommendations_ann(
        seed_id=seed_id,
        id_to_item=id_to_item,
        embeddings=embeddings,
        k=args.k,
        cache_path=cache_path,
    )

    seed_title = id_to_item.get(seed_id, {}).get("Title", "N/A")
    print(f"Seed Item: '{seed_title}' (Id: {seed_id})")
    print("Top recommendations:")
    for r in recs:
        genres_val = r.get("Genres", [])
        genres_str = ", ".join(genres_val) if isinstance(genres_val, list) else str(genres_val)
        print(f"- Id: {r['Id']:<10} | Title: {r['Title']:<40} | Genres: {genres_str:<30} | Score: {r['Score']:.5f}")

    if args.save:
        save_embeddings(args.save, embeddings)
        print(f"Saved embeddings to: {args.save}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        main()
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"\nElapsed time: {elapsed:.3f}s")
