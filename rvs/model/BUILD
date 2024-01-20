# RVS: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'util',
    srcs = ['util.py'],    
    deps = [
        "//rvs/geo:util",

],
)



py_binary(
    name = 'baselines',
    main = 'baselines.py',
    srcs = ['baselines.py'],
    deps = [
          ":datasets",
          "//rvs/geo:osm",
          "//rvs/geo:util",
          "//rvs/model:util",
          "//rvs/geo:regions",
          "//rvs/evals:utils"
    ],
)


py_binary(
    name = 'datasets',
    main = 'datasets.py',
    srcs = ['datasets.py'],
    deps = [
          "//rvs/geo:util",
          "//rvs/geo:regions",
          "//rvs/evals:utils",
          ":dataset_item",
          ":util"


    ],
)

py_library(
    name = 'dataset_item',
    srcs = ['dataset_item.py'],
    deps = [
        "//rvs/geo:util",
        ":util"


],

)



py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    python_version = "PY3",
    deps = [":util"],
)
