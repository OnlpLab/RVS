package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = 'observe',
    srcs = ['observe.py'],
)

py_test(
    name = "observe_test",
    srcs = ["observe_test.py"],
    python_version = "PY3",
    deps = [
        ":observe",
        "//rvs/data/wikidata:item",
        "//rvs/data/wikidata:query",
        '//rvs/geo:regions',
    ],
)

py_library(
    name = 'speak',
    srcs = ['speak.py'],
    deps = [
        '//rvs/geo:directions',
    ]
)


py_test(
    name = "speak_test",
    srcs = ["speak_test.py"],
    python_version = "PY3",
    deps = [
        ":speak",
        '//rvs/geo:directions',
    ],
)

py_test(
    name = "templates_test",
    srcs = ["templates_test.py"],
    python_version = "PY3",
    deps = [
        ":templates",
        '//rvs/geo:geo_item',
        '//rvs/geo:util',

    ],
)

py_library(
    name = 'prompt_LLM_gen_functions',
    srcs = ['prompt_LLM_gen_functions.py'],
    deps = [
                '//rvs/geo:geo_item',

        ],
)

py_binary(
    name = 'generate_synth',
    main = 'generate_synth.py',
    srcs = ['generate_synth.py'],
    deps = [
        ":templates",
        '//rvs/geo:geo_item',
        "//rvs/geo:walk",


    ],
)

py_binary(
    name = 'generate_synth_prompting',
    main = 'generate_synth_prompting.py',
    srcs = ['generate_synth_prompting.py'],
    deps = [
        ":prompt_LLM_gen_functions",
        '//rvs/geo:walk',

    ],
)

py_library(
    name = 'templates',
    srcs = ['templates.py'],
        deps = [
        '//rvs/geo:geo_item',
        '//rvs/geo:walk',
         '//rvs/geo:util',

    ],
)
