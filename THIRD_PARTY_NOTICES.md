# THIRD_PARTY_NOTICES.md

This repository includes or adapts third-party open-source code.

本仓库包含或改写了第三方开源代码。

---

## 1. 3DGS.cpp

- Upstream repository / 上游仓库: <https://github.com/shg8/3DGS.cpp>
- Upstream project name / 上游项目名称: **3DGS.cpp**
- Upstream license / 上游许可证: **GNU Lesser General Public License v2.1 (LGPL-2.1)**, with the upstream README stating that the main project is licensed under LGPL.
- Local usage / 本仓库中的使用方式: Portions of the Gaussian Splatting Vulkan implementation in this repository were derived from or adapted from `3DGS.cpp`.

### Notice / 说明

Please retain all relevant attribution, copyright notices, and license notices when redistributing this repository or modifying the affected source files.

如果你继续分发本仓库，或继续修改相关源文件，请保留对应的版权说明、致谢说明与许可证信息。

---

## 2. Other third-party libraries mentioned by the upstream project

The upstream `3DGS.cpp` README lists the following third-party libraries and licenses:

- **GLM** — MIT License
- **args.hxx** — MIT License
- **spdlog** — MIT License
- **ImGUI** — MIT License
- **Vulkan Memory Allocator** — MIT License
- **VkRadixSort** — MIT License
- **implot** — MIT License
- **glfw** — zlib/libpng license
- **libenvpp** — Apache License 2.0

If any of these components are redistributed as part of this repository, their corresponding license obligations should also be preserved.

上游 `3DGS.cpp` README 还列出了若干第三方库及其许可证。如果这些组件随本仓库一起分发，也应保留其对应的许可证要求。

---

## 3. Recommendation for downstream reuse

If you reuse this repository or redistribute modified versions of it, it is recommended that you preserve:

- `README.md`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- any copyright / attribution headers retained in source files

如果你复用本仓库，或分发其修改版本，建议保留：

- `README.md`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- 以及源文件中保留的版权/致谢头部说明
