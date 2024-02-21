# 如何利用 MkDocs 搭建个人博客

## 配置 GitHub Pages

1. 创建一个新的仓库，仓库名为 `username.github.io`，其中 `username` 是你的 GitHub 用户名
2. 这样你的博客就可以通过 `https://username.github.io` 访问了。网页的根目录就是你的整个仓库，可以通过修改仓库文件来制作自己的网页

同时，也可以通过现成框架快速搭建自己的博客，例如 MkDocs

## 配置 MkDocs

- 安装 MkDocs

```bash
pip install mkdocs-material
```

- 使用 MkDocs

```bash
# 在当前目录创建一个新的 MkDocs 项目
mkdocs new .
# 预览，访问地址 http://localhost:8000
mkdocs serve
```

## 一种博客的配置方法: 利用导航栏配置博客

- 在 `mkdocs.yml` 中配置 `site_name`、`site_description`、`site_author` 等信息
- 在 `docs` 目录下创建 `.md` 文件，即博客的内容
- 在 `mkdocs.yml` 中配置 `nav`，指定导航栏的内容。并通过 key-value 的形式指定导航栏的名称和对应的文件路径

```yml
nav:
  - Home: index.md
  - About: about.md
  - Blog:
      - "2021": 2021/index.md
      - "2020": 2020/index.md
```

- 在 `mkdocs.yml` 中配置 `theme`，指定主题

```yml
theme:
  name: material
```

## 部署到 GitHub Pages

可以通过 GitHub Actions 实现自动部署，只需要在仓库的 `.github/workflows` 目录下创建一个 `.yml` 文件，例如 `ci.yml`，并在其中配置自动部署的流程

```yml
name: ci
on:
  push:
    branches:
      - master
      - main
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure Git Credentials
        run: |
          git config user.name github-actions[bot]
          git config user.email 41898282+github-actions[bot]@users.noreply.github.com
      - uses: actions/setup-python@v5
        with:
          python-version: 3.x
      - run: echo "cache_id=$(date --utc '+%V')" >> $GITHUB_ENV
      - uses: actions/cache@v4
        with:
          key: mkdocs-material-${{ env.cache_id }}
          path: .cache
          restore-keys: |
            mkdocs-material-
      - run: pip install mkdocs-material
      - run: mkdocs gh-deploy --force
```

以上这个配置文件的作用是，当 `master` 或 `main` 分支有新的提交时，自动部署到 GitHub Pages。部署的网页会在 gh-pages 分支上。因此，需要在仓库的 `Settings` -> `Pages` 中配置 `gh-pages` 分支作为网站的源

![deploy branch setting](deploy_setting_light.png#only-light)
![deploy branch setting](deploy_setting_dark.png#only-dark)
