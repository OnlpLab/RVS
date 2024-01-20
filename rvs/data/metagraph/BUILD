package(
    default_visibility = ["//visibility:public"],
)

py_binary(
    name = 'create_graph_embedding',
    srcs = ['create_graph_embedding.py'],
    deps = [
        "//rvs/geo:regions",
        ":utils",
    ],
)

py_library(
    name = 'utils',
    srcs = ['utils.py'],
    deps = [
    "//rvs/data/wikidata:query",
    "//rvs/geo:regions",
    "//rvs/geo:util",
    "//rvs/geo/map_processing:map_structure",

    ],
)


py_test(
    name = "graph_embedding_test",
    srcs = ["graph_embedding_test.py"],
    python_version = "PY3",
    deps = [
        "//rvs/geo:regions",
        ":utils",
        '//rvs/geo/map_processing:map_structure'
    ],
)