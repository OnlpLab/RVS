package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'walk',
    srcs = ['walk.py'],
    deps = [
        ':util',
        ':geo_item',
        ':osm',
        '//rvs/geo/map_processing:map_structure',
        '//rvs/geo:regions'

    ],

)

py_library(
    name = 'osm',
    srcs = ['osm.py'],
)

py_library(
    name = 'geo_item',
    srcs = ['geo_item.py'],
)


py_library(
    name = 'util',
    srcs = ['util.py'],
)

py_library(
    name = 'regions',
    srcs = ['regions.py'],
)

py_test(
    name = "walk_test",
    srcs = ["walk_test.py"],
    python_version = "PY3",
    deps = [
        ":walk",
        ':geo_item',
        "//rvs/geo/map_processing:map_structure",
        '//rvs/geo:regions'
    ],
		
)

py_library(
    name = 'directions',
    srcs = ['directions.py'],
    deps = [
        ':walk',
        ':util',
        ':geo_item',
        "//rvs:logger",
        "//rvs/geo/map_processing:map_structure",
        ],
)

py_binary(
    name = 'sample_poi',
    main = 'sample_poi.py',
    srcs = ['sample_poi.py'],
    deps = [
        ':walk',
        ':util',
        '//rvs/geo:geo_item',
        "//rvs:logger",
        '//rvs/geo:regions'
        ],
)