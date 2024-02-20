---
draft: false
date: 2024-02-20
categories:
  - Web Development
---

# 如何利用 MkDocs 搭建个人博客

## 安装 MkDocs

```bash
pip install mkdocs-material
```

## 创建一个新的 MkDocs 项目

```bash
# 在当前目录创建一个新的 MkDocs 项目
mkdocs new .
# 预览，访问地址 http://localhost:8000
mkdocs serve
```

## 配置博客

修改 `mkdocs.yml` 文件，添加 `blog` 插件

```yaml
plugins:
  - blog
```

博客文章放在 `docs/blog/posts` 目录下，所有的博客开头都需要添加元数据，例如：

```markdown
---
draft: true
date: 2024-02-20
categories:
  - Web Development
---

# 如何利用 MkDocs 搭建个人博客
```
