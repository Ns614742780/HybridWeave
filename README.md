# HybridWeave

> English and Chinese versions are provided in the same document for convenience.  
> 本 README 同时提供英文与中文说明，便于直接用于公开仓库。

---

## English

### Overview

**HybridWeave** is a Vulkan-based real-time hybrid renderer for **mesh + 3D Gaussian Splatting (3DGS)** scenes.  
This repository contains the research code accompanying our submitted paper:

**HybridWeave: Coordinated Optimization for Hybrid Mesh–3DGS Real-Time Rendering**

The current implementation is intended primarily for:

- research reproduction,
- qualitative testing,
- method demonstration,
- and further experimentation on hybrid Mesh–3DGS rendering.

It combines:

- a **glTF mesh rendering branch**,
- a **Vulkan-based 3DGS branch**,
- and a **hybrid composition stage** under a shared camera.

---

### Features

- Real-time hybrid rendering of **mesh + 3DGS** content
- Shared camera for both rendering branches
- Vulkan-based Gaussian Splatting pipeline
- glTF-based mesh rendering pipeline
- Hybrid composition for mixed-scene experiments
- Easy customization of camera, mesh transform, and scene paths

---

### Project Status

This repository is a **research code release** and a **paper companion implementation**.

It is mainly intended for:

- reproducing the rendering pipeline in the paper,
- testing hybrid scenes,
- running qualitative demos,
- and serving as a baseline for future work.

It is **not** currently packaged as a production-ready renderer.

---

### Repository Contents and Minimal Assets

This repository includes:

- the source code,
- one default glTF mesh asset,
- one default HDR environment map,
- and the default runtime configuration used by the sample setup.

The default Gaussian scene file

```text
assets/pointclouds/bicycle.ply
```

is **not stored directly in the Git repository**, because it is too large for convenient repository distribution.

Instead, it is provided separately through the **GitHub Releases** page.

After downloading the file, place it at:

```text
assets/pointclouds/bicycle.ply
```

so that the default configuration in `main` can run without further path changes.

---

### Requirements

- **Windows**
- **Visual Studio 2022**
- **Vulkan SDK**
- A Vulkan-capable GPU and driver

Please make sure the Vulkan SDK is installed correctly and that Visual Studio can locate the required headers and libraries.

> **Note**  
> This repository does **not** bundle the Vulkan SDK.  
> Please install the Vulkan SDK separately before building the project, and make sure the `VULKAN_SDK` environment variable is configured correctly.

---

### Build and Run

This project is intended to be built with **Visual Studio 2022**.

#### Build

Open the solution in Visual Studio 2022 and build the project.

Before opening the solution in Visual Studio 2022, please make sure that:

- the Vulkan SDK has been installed,
- the `VULKAN_SDK` environment variable is available,
- and the required Vulkan tools such as `glslc` can be found under the installed SDK.

#### Run

After building, simply run:

```bash
HybridWeave.exe
```

No additional command-line setup is required for the default configuration.

---

### Quick Customization

This repository is structured so that you can quickly adapt the code to your own assets and scenes.

#### 1. Change the camera parameters

If you want to start with a different camera position, rotation, or projection parameters, edit the camera initialization in `Renderer.h`:

```cpp
// if you want to start with a different camera position, rotation or projection parameters, change the values here
Camera camera{
    glm::vec3(3.600350f, -2.643269f, 5.880525f),
    glm::quat(-0.061977f, -0.266743f, 0.217666f, -0.936818f),
    45,
    0.1f,
    3000.0f
};
```

This is usually the first place to adjust when loading a new scene or when the initial view is not appropriate for the scene scale.

#### 2. Change the mesh model transform

If you want to change the position, scale, or orientation of the glTF model, edit the root transform in `GltfRenderPass.cpp`:

```cpp
void GltfRenderPass::initialize()
{
    spdlog::debug("GltfRenderPass::initialize() scene={}", scenePath);

    loader = std::make_unique<GltfLoaderVulkan>(context);
    GltfLoaderVulkan::RootTransform t;

    // bicycle
    // if you want to transform the model, you can change the values below or set them to identity
    t.translate = { -2.5f, 2.0f, -1.0f };
    t.scale = { 1.0f, 1.0f, 1.0f };
    t.rotate = glm::angleAxis(glm::radians(180.0f), glm::vec3(1, 0, 0));
```

This is useful when aligning the mesh model with the 3DGS scene in hybrid rendering experiments.

#### 3. Change the scene files

If you want to test with your own assets, replace the default scene paths in `main`:

```cpp
// you can replace the scene files here to test with your own scenes, just make sure the camera settings in RendererConfiguration are appropriate for the scene scale
config.sceneGS = "assets/pointclouds/bicycle.ply";

// you can replace the glTF scene with your own model, but make sure it has a PBR material and is not too heavy for testing
// (we recommend using a single object with less than 100k triangles for testing, and you can use the camera settings in RendererConfiguration to adjust the view)
config.sceneGLTF = "assets/models/tree/tree_small_02_4k.gltf";

// you can replace the IBL environment map here, just make sure it's an HDR image and adjust the camera settings in RendererConfiguration
// if the scene is too dark or too bright with the new environment
config.iblHdrPath = "assets/envs/overcast_soil_puresky_4k.hdr";
```

When replacing assets, please make sure that:

- the Gaussian scene file exists and is readable,
- the glTF model uses reasonable PBR materials,
- the mesh is not excessively heavy for initial testing,
- the HDR environment map is valid,
- and the camera parameters are adjusted to match the scene scale.

---

### Notes for Testing Your Own Scenes

For more stable first-time testing, we recommend:

- starting with a **single glTF object**,
- keeping the mesh at a **moderate triangle count**,
- checking the camera placement first if the scene appears empty,
- and adjusting the mesh transform before changing more advanced logic.

If the scene appears too dark, too bright, too small, or visually misaligned, the first places to check are:

1. camera parameters in `Renderer.h`,
2. mesh transform in `GltfRenderPass.cpp`,
3. scene file paths in `main`.

---

### Upstream Acknowledgment

This project includes code derived from and adapted from the following open-source project:

- **3DGS.cpp** — <https://github.com/shg8/3DGS.cpp>

We gratefully acknowledge the authors of `3DGS.cpp` for releasing a Vulkan-based Gaussian Splatting implementation that has been valuable for research and engineering reference.

---

### Citation

If you use this repository in academic work, please cite our paper:

```bibtex
@unpublished{hybridweave2026,
  title={HybridWeave: Coordinated Optimization for Hybrid Mesh--3DGS Real-Time Rendering},
  author={NIE, Shuai and WANG, Chongwen},
  note={Manuscript under review},
  year={2026}
}
```

If your work also relies on the Gaussian Splatting implementation lineage from `3DGS.cpp`, please also cite the upstream project and the original 3D Gaussian Splatting paper where appropriate.

---

### License

This repository is distributed under the **GNU Lesser General Public License v2.1 or later (LGPL-2.1-or-later)**.

Please see:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

for the complete licensing and attribution information.

If you redistribute this repository, please preserve:

- this README,
- the `LICENSE` file,
- third-party notices,
- and any copyright / attribution notices retained in the source code.

---

### Disclaimer

This repository is released for research and educational purposes and is provided **as is**, without warranty of any kind. See the `LICENSE` file for details.

---

## 中文

### 项目简介

**HybridWeave** 是一个基于 Vulkan 的 **Mesh + 3D Gaussian Splatting (3DGS)** 实时混合渲染研究原型。  
本仓库包含会议论文配套代码：

**HybridWeave: Coordinated Optimization for Hybrid Mesh–3DGS Real-Time Rendering**

当前实现主要用于：

- 论文方法复现，
- 定性实验与演示，
- 混合渲染流程验证，
- 以及后续研究扩展。

该实现主要包含：

- **基于 glTF 的网格渲染分支**，
- **基于 Vulkan 的 3DGS 渲染分支**，
- **在共享相机下进行混合合成的渲染流程**。

---

### 主要特性

- 支持 **Mesh + 3DGS** 的实时混合渲染
- 两个分支共享相机参数
- 基于 Vulkan 的 Gaussian Splatting 渲染流程
- 基于 glTF 的 Mesh 渲染流程
- 支持混合场景实验与可视化组合
- 方便修改相机、模型变换和场景路径

---

### 项目定位

本仓库是一个**研究代码仓库**，同时也是**论文配套实现**。

它主要用于：

- 复现论文中的渲染流程，
- 测试混合场景，
- 进行定性可视化演示，
- 作为后续研究的基线实现。

目前它**不是**一个面向生产环境打包完善的通用渲染器。

---

### 仓库内容与最小资源说明

本仓库包含：

- 源代码，
- 一个默认的 glTF 网格资源，
- 一个默认的 HDR 环境贴图，
- 以及与示例配置对应的默认运行路径设置。

默认 3DGS 点云文件：

```text
assets/pointclouds/bicycle.ply
```

**没有直接放在 Git 仓库中**，因为该文件体积较大，不适合直接随仓库分发。

该文件通过 **GitHub Releases** 页面单独提供。

下载后，请将其放到：

```text
assets/pointclouds/bicycle.ply
```

这样就可以直接使用 `main` 中的默认路径配置运行，无需额外修改代码。

---

### 环境要求

- **Windows**
- **Visual Studio 2022**
- **Vulkan SDK**
- 支持 Vulkan 的 GPU 与驱动

请确保 Vulkan SDK 已正确安装，并且 Visual Studio 可以找到相应的头文件与库文件。

> **说明**  
> 本仓库**不附带** Vulkan SDK。  
> 在编译项目前，请先自行安装 Vulkan SDK，并确保 `VULKAN_SDK` 环境变量已正确配置。

---

### 编译与运行

本项目推荐使用 **Visual Studio 2022** 进行编译。

在使用 Visual Studio 2022 打开解决方案之前，请先确认：

- 已正确安装 Vulkan SDK；
- `VULKAN_SDK` 环境变量可用；
- 已安装的 Vulkan SDK 中包含 `glslc` 等所需工具。

#### 编译

使用 Visual Studio 2022 打开解决方案并完成编译。

#### 运行

编译完成后，直接运行：

```bash
HybridWeave.exe
```

默认测试配置下，不需要额外命令行参数。

---

### 快速修改说明

为了便于测试不同场景，本仓库将常见可调项集中在几个比较直接的位置。

#### 1. 修改相机参数

如果你希望从不同的相机位置、朝向或投影参数开始运行，可以在 `Renderer.h` 中修改相机初始化：

```cpp
// if you want to start with a different camera position, rotation or projection parameters, change the values here
Camera camera{
    glm::vec3(3.600350f, -2.643269f, 5.880525f),
    glm::quat(-0.061977f, -0.266743f, 0.217666f, -0.936818f),
    45,
    0.1f,
    3000.0f
};
```

当你更换场景后，如果初始视角不合适，通常应优先修改这里。

#### 2. 修改 glTF 模型的变换

如果你希望调整 glTF 模型的位置、缩放或朝向，可以在 `GltfRenderPass.cpp` 中修改根变换：

```cpp
void GltfRenderPass::initialize()
{
    spdlog::debug("GltfRenderPass::initialize() scene={}", scenePath);

    loader = std::make_unique<GltfLoaderVulkan>(context);
    GltfLoaderVulkan::RootTransform t;

    // bicycle
    // if you want to transform the model, you can change the values below or set them to identity
    t.translate = { -2.5f, 2.0f, -1.0f };
    t.scale = { 1.0f, 1.0f, 1.0f };
    t.rotate = glm::angleAxis(glm::radians(180.0f), glm::vec3(1, 0, 0));
```

这对于将 mesh 模型与 3DGS 场景对齐尤其有用。

#### 3. 修改场景文件路径

如果你希望替换为自己的场景资源，可以在 `main` 中修改默认路径：

```cpp
// you can replace the scene files here to test with your own scenes, just make sure the camera settings in RendererConfiguration are appropriate for the scene scale
config.sceneGS = "assets/pointclouds/bicycle.ply";

// you can replace the glTF scene with your own model, but make sure it has a PBR material and is not too heavy for testing
// (we recommend using a single object with less than 100k triangles for testing, and you can use the camera settings in RendererConfiguration to adjust the view)
config.sceneGLTF = "assets/models/tree/tree_small_02_4k.gltf";

// you can replace the IBL environment map here, just make sure it's an HDR image and adjust the camera settings in RendererConfiguration
// if the scene is too dark or too bright with the new environment
config.iblHdrPath = "assets/envs/overcast_soil_puresky_4k.hdr";
```

更换资源时建议注意：

- 3DGS 场景路径必须有效；
- glTF 模型最好具备合理的 PBR 材质；
- 初始测试时 mesh 不宜过重；
- IBL 环境图应为合法 HDR 文件；
- 相机参数应与场景尺度匹配。

---

### 使用你自己的场景时的建议

为了更稳定地完成首次测试，建议：

- 先使用**单个 glTF 模型**进行验证；
- 初始测试时保持 **适中的三角形数量**；
- 如果画面什么都看不到，优先检查相机位置；
- 先调模型变换，再调更复杂的渲染逻辑。

如果结果表现为过暗、过亮、过小或者位置不对齐，建议优先检查：

1. `Renderer.h` 中的相机参数；
2. `GltfRenderPass.cpp` 中的模型变换；
3. `main` 中的场景文件路径。

---

### 上游项目致谢

本项目包含了来自以下开源项目的派生/改写代码：

- **3DGS.cpp** — <https://github.com/shg8/3DGS.cpp>

感谢 `3DGS.cpp` 的作者公开其基于 Vulkan 的 Gaussian Splatting 实现，为本研究提供了重要的工程参考。

---

### 引用方式

如果你在学术工作中使用了本仓库，请引用我们的论文：

```bibtex
@unpublished{hybridweave2026,
  title={HybridWeave: Coordinated Optimization for Hybrid Mesh--3DGS Real-Time Rendering},
  author={NIE, Shuai and WANG, Chongwen},
  note={Manuscript under review},
  year={2026}
}
```

如果你的工作也使用或继承了 `3DGS.cpp` 的 Gaussian Splatting 实现脉络，请同时引用上游项目和原始 3D Gaussian Splatting 论文。

---

### 许可证说明

本仓库采用 **GNU Lesser General Public License v2.1 or later（LGPL-2.1-or-later）** 进行发布。

完整许可与第三方说明请见：

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

如果你继续分发本仓库，请保留：

- 本 README，
- `LICENSE` 文件，
- 第三方说明文件，
- 以及源代码中保留的版权/致谢说明。

---

### 免责声明

本仓库仅用于研究与教学目的，按 **现状（as is）** 提供，不附带任何明示或暗示担保。详细内容请见 `LICENSE` 文件。
