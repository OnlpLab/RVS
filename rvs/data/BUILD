package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'extract',
    srcs = ['extract.py'],
    deps = [
        ":wikigeo",
        "//rvs/data/wikidata:item",
        "//rvs/data/wikidata:query",
        "//rvs/data/wikipedia:item",
        "//rvs/data/wikipedia:query",
        "//rvs/data/wikidata:info_item",
        '//rvs/geo/map_processing:map_structure'
    ],
)

py_library(
    name = 'wikigeo',
    srcs = ['wikigeo.py'],
)

py_library(
    name = 'osm_item',
    srcs = ['osm_item.py'],
)

py_test(
    name = "extract_test",
    srcs = ["extract_test.py"],
    python_version = "PY3",
    deps = [
        ":extract",
        ":wikigeo",
        ":osm_item",
        "//rvs/data/wikidata:item",
        "//rvs/data/wikidata:query",
        "//rvs/data/wikipedia:item",
        "//rvs/data/wikipedia:query",
        "//rvs/data/wikidata:info_item",
        '//rvs/geo/map_processing:map_structure'
    ],
)

py_binary(
    name = 'create_wikigeo_dataset',
    srcs = ['create_wikigeo_dataset.py'],
    deps = [
        ':extract',
        ":osm_item",
        "//rvs/data/wikidata:item",
        "//rvs/data/wikidata:query",
        "//rvs/data/wikipedia:item",
        "//rvs/data/wikipedia:query",
        "//rvs/data:wikigeo"
    ],
)