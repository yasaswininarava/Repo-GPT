from chat_github.chunking import ChunkRecord
from chat_github.github_loader import github_blob_url
from chat_github.index_store import build_index, cache_directory, load_index, save_index, search
from tests.fakes import HashEmbedder


def _index(tmp_path=None):
    embedder = HashEmbedder()
    chunks = [
        ChunkRecord(
            path="src/alpha.py",
            start_line=1,
            end_line=2,
            text="def uniquealpha():\n    return 1",
            commit="aaa111",
            owner="octo",
            repo="demo",
        ),
        ChunkRecord(
            path="src/beta.py",
            start_line=4,
            end_line=5,
            text="def uniquebeta():\n    return 2",
            commit="aaa111",
            owner="octo",
            repo="demo",
        ),
    ]
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    index = build_index(
        owner="octo",
        repo="demo",
        commit="aaa111",
        default_branch="main",
        chunks=chunks,
        vectors=vectors,
        embedding_model=embedder.name,
        max_input_tokens=embedder.max_input_tokens,
        files_indexed=2,
        skipped_summary={"secret_file": 1},
        incomplete=False,
        incomplete_reasons=[],
        notices=[],
    )
    if tmp_path is not None:
        directory = tmp_path / "index"
        save_index(index, directory)
        index = load_index(directory)
    return index, embedder


def test_search_maps_back_to_the_stored_source():
    index, embedder = _index()
    hits = search(index, embedder.embed_query("where is uniquebeta defined"), top_k=1)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.source_id == "S1"
    assert hit.chunk.path == "src/beta.py"
    assert hit.chunk.start_line == 4
    assert hit.chunk.end_line == 5
    assert hit.chunk.url == github_blob_url(
        "octo",
        "demo",
        "aaa111",
        "src/beta.py",
        4,
        5,
    )
    assert "https://github.com/octo/demo/blob/aaa111/src/beta.py#L4-L5" == hit.chunk.url


def test_saved_index_reloads_the_same_source_mapping(tmp_path):
    index, embedder = _index(tmp_path)
    hits = search(index, embedder.embed_query("uniquebeta"), top_k=1)
    assert hits[0].chunk.text.startswith("def uniquebeta")
    assert hits[0].chunk.url.endswith("src/beta.py#L4-L5")
    assert index.files_indexed == 2
    assert index.skipped_summary["secret_file"] == 1
    assert index.loaded_from_cache is True


def test_cache_path_changes_when_the_commit_changes(tmp_path):
    first = cache_directory("octo", "demo", "commit-a", embedding_model="hash-test", max_input_tokens=64, cache_root=tmp_path)
    second = cache_directory("octo", "demo", "commit-b", embedding_model="hash-test", max_input_tokens=64, cache_root=tmp_path)
    assert first != second
    assert first.parent == second.parent
