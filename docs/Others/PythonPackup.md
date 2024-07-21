# 打包 Python 项目并上传至 PyPI

## 1. 创建自己的 Package

Python 的 Package 是一个文件夹，其中包含一个 `__init__.py` 文件，用于标识该文件夹为一个 Package。

在`__init__.py`中可以定义一些初始化的操作，也可以定义一些函数、类等。

```python
from . import module1
from . import module2
from module1 import func1
```

在 Package 中可以包含多个模块，也可以包含子 Package。

## 2. 打包准备和配置

一个示例的项目结构如下：

```
my_project/
│
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── module1.py
│       └── module2.py
│
├── tests/
│   └── test_module1.py
│
├── setup.py
├── pyproject.toml
└── README.md
```

其中 setup.py 描述了如何打包这个 Package，以及如何上传到 PyPI，而 pyproject.toml 描述了打包使用的工具。

### 2.1. setup.py

下面是一个示例的 setup.py 文件：

```python
from setuptools import setup, find_packages

setup(
    name="my_package",
    version="0.1.0",
    description="A description of my package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="youremail@example.com",
    url="https://github.com/yourusername/my_package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        # List your project's dependencies here.
        # e.g. 'requests>=2.24.0',
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
        ],
    },
)

```

### 2.2. pyproject.toml

由于使用 setuptools 进行打包，打包的配置大多在 setup.py 中，因此 pyproject.toml 中的配置较少：

```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"
```

## 3. 打包 & 上传

### 3.1. 打包

在项目根目录下运行：

```shell
python setup.py sdist bdist_wheel
```

这会在 dist/ 目录下生成一个 tar.gz 文件和一个 .whl 文件。

### 3.2. 上传

在上传之前，需要先注册一个 PyPI 账号，然后创建一个 API token。

[How to upload a package to PyPI](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives)

这个教程中有详细的步骤，其中描述的是如何上传至 Test PyPI（一个完全独立的测试环境）。如果要上传至真正的 PyPI，可以直接执行下面的命令。

```shell
pip install twine
twine upload dist/*
```
