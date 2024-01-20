# RVS: Geographic vision-and-language navigation

licenses(["notice"])

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)


py_binary(
    name = 'model_trainer',
    main = 'model_trainer.py',
    srcs = ['model_trainer.py'],
    deps = [
        '//rvs/model/text:train',
        '//rvs/model:datasets',
        '//rvs/model:dataset_item',
        "//rvs/model:util",
        "//rvs/geo:util",
        ':models'        

    ],
)

py_binary(
    name = 'model_trainer_multitask',
    main = 'model_trainer_multitask.py',
    srcs = ['model_trainer_multitask.py'],
    deps = [
        '//rvs/model/text:train',
        '//rvs/model:datasets',
        '//rvs/model:dataset_item',
        "//rvs/model:util",
        "//rvs/geo:util",
        ':models'        

    ],
)



py_binary(
    name = 'models',
    main = 'models.py',
    srcs = ['models.py'],
    deps = [
        "//rvs/model:util",

],

)

py_library(
    name = 'train',
    srcs = ['train.py'],    
    deps = [
        "//rvs/model:util",
        "//rvs/evals:utils",

],
)