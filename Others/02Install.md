<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 本地部署

## Sphinx 环境安装

大模型系统项目部署在 Github 是依赖于 sphinx 工具实现的。因此我们首先要安装 sphinx。在 MacOS 中，可以使用 Homebrew 、 MacPorts 或者 Anaconda 之类的 Python 发行版安装 Sphinx。

```bash
brew install sphinx-doc
```

接着通过 `pip` 安装 `sphinx-book-theme`：

```bash
pip install sphinx-book-theme
```

然后，在 Sphinx 配置（`conf.py`）中激活主题：

```
...
html_theme = "sphinx_book_theme"
...
``` 

这将为您的文档激活 `sphinx_book_theme` 图书主题。

## 写入内容与图片

因为《大模型系统》的内容都存放在 https://github.com/Infrasys-AI/AIInfra/ 地址上，因此需要通过 github desktop 或者 git clone http 的方式拉取下来到本地。

> 因为网络不稳定的问题，建议翻墙或者直接使用 github desktop 软件应用下载，使其支持断点下载项目。

接着进入 AIInfra 目录下的 `build_books` 文件，并修改里面的源目录地址 `xxxxx/AIInfra` 和目标构建本地部署内容的地址 `xxxxx/AIInfra_BOOK`。

```python
target_dir1 = '/xxxxx/AIInfra/02Hardware'
target_dir2 = '/xxxxx/AIInfra/03Compiler'
target_dir3 = '/xxxxx/AIInfra/04Inference'
target_dir4 = '/xxxxx/AIInfra/05Framework'
dir_paths = '/xxxxx/AIInfra_BOOK/source/'

getallfile(target_dir1)
getallfile(target_dir2)
getallfile(target_dir3)
getallfile(target_dir4)
```

最后执行 `build_books/create_dir.py` 文件，实现写入本地部署的内容与图片。

```bash
python create_dir.py
```

## 编译 HTML 版本

在编译前先去到需要编译的目录，所有的编译命令都在这个文件目录内执行。

```bash
cd AIInfra_BOOK
make html
```

生成的 html 会在`build/html`，打开目录下的 html 文件即可进入本地部署环境。

此时我们将编译好的 html 整个文件夹下的内容拷贝至 xxxxxx.github.io 发布。

需要注意的是 docs(AIInfra_BOOK) 目录下的 /source/index.md 不要删除了，不然网页无法检索渲染。

## 配置文件与代码

大模型系统在 Sphinx 配置（`conf.py`）中的全部配置内容：

<!--
```python
# -- Project information -----
import os
from urllib.request import urlopen
from pathlib import Path

project = "AIInfra & AIInfra (大模型系统原理)"
language = "cn"  # For testing language translations
master_doc = "index"

# -- General configuration ------
extensions = [
    "ablog",
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_examples",
    "sphinx_tabs.tabs",
    "sphinx_thebe",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
    "sphinxext.opengraph",
    "sphinx.ext.todo",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "pst": ("https://pydata-sphinx-theme.readthedocs.io/en/latest/", None),
}
nitpick_ignore = [
    ("py:class", "docutils.nodes.document"),
    ("py:class", "docutils.parsers.rst.directives.body.Sidebar"),
]

suppress_warnings = ["myst.domains", "ref.ref"]

numfig = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]

# -- Options for HTML output ----
html_theme = "sphinx_book_theme"
html_logo = "_static/logo-wide.svg"
html_title = "AI System"
html_copy_source = True
html_favicon = "_static/logo-square.svg"
html_last_updated_fmt = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
nb_execution_mode = "cache"
thebe_config = {
    "repository_url": "https://github.com/binder-examples/jupyter-stacks-datascience",
    "repository_branch": "master",
}

html_theme_options = {
    "path_to_docs": "",
    "repository_url": "https://github.com/chenzomi12/chenzomi12.github.io/",
    "repository_branch": "master",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "deepnote_url": "https://deepnote.com/",
        "notebook_interface": "jupyterlab",
        "thebe": True,
    },
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 3,
    "logo": {
        "image_dark": "_static/logo-wide.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Infrasys-AI/AIInfra",
            "icon": "fa-brands fa-github",
        }, {
            "name": "Youtube",
            "url": "https://www.youtube.com/@ZOMI666",
            "icon": "fa-brands fa-youtube"
        }, {
            "name": "Blibili",
            "url": "https://space.bilibili.com/517221395",
            "icon": "fa-brands fa-bilibili",
        }
    ],
}

# sphinxext.opengraph
ogp_social_cards = {
    "image": "_static/logo-square.png",
}

# # -- ABlog config ----
blog_path = "reference/blog"
blog_post_pattern = "reference/blog/*.md"
blog_baseurl = "https://sphinx-book-theme.readthedocs.io"
fontawesome_included = True
post_auto_image = 1
post_auto_excerpt = 2
nb_execution_show_tb = "READTHEDOCS" in os.environ
bibtex_bibfiles = ["references.bib"]
# To test that style looks good with common bibtex config
bibtex_reference_style = "author_year"
bibtex_default_style = "plain"
numpydoc_show_class_members = False  # for automodule:: urllib.parse stub file issue
linkcheck_ignore = [
    "http://someurl/release",  # This is a fake link
    "https://doi.org",  # These don't resolve properly and cause SSL issues
]

def setup(app):
    if not os.environ.get("READTHEDOCS") and not os.environ.get("GITHUB_ACTIONS"):
        app.add_css_file(
            "https://assets.readthedocs.org/static/css/readthedocs-doc-embed.css"
        )
        app.add_css_file("https://assets.readthedocs.org/static/css/badge_only.css")
        app.add_js_file("rtd-data.js")
        app.add_js_file(
            "https://assets.readthedocs.org/static/javascript/readthedocs-doc-embed.js",
            priority=501,
        )
```
-->

需要渲染的主页链接 `index.md` 跟 `conf.py` 一样放在 source 文件目录下：

<!--
```md
---
title: AIInfra & AIInfra 
---

# 课程目录内容

```{toctree}
:maxdepth: 1
:caption: === 一.大模型系统概述 ===

01Introduction/README
01Introduction/01Present
01Introduction/02Develop
01Introduction/03Architecture
01Introduction/04Sample
```

Thanks you!!!
```
-->