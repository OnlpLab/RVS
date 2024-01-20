package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'item',
    srcs = ['item.py'],
)

py_library(
    name = 'info_item',
    srcs = ['info_item.py'],
)

py_library(
    name = 'query',
    srcs = ['query.py'],
    deps = [
        "//rvs/geo:util",
        '//rvs/geo:regions',
    ]
)

py_test(
    name = "query_test",
    srcs = ["query_test.py"],
    python_version = "PY3",
    deps = [
        ":query",
        "//rvs/geo:util",
        '//rvs/geo:regions',
    ],
)

py_binary(
    name = 'extract_geofenced_wikidata_items',
    main = 'extract_geofenced_wikidata_items.py',
    srcs = ['extract_geofenced_wikidata_items.py'],
    deps = [
      ':query',
      '//rvs/geo:regions',
    ],
)