# DATA1030 Weather Project — 项目框架

这是用于小组与 TA 讨论的天气预测项目基础框架。当前交付包括配置、可复用模块和**未执行、无输出的 [Weather_Project.ipynb](Weather_Project.ipynb)**；数据获取、训练、分析和结果图均留待方案确定后手动执行。

## 目前的讨论方案

- 七个城市的机场坐标已写入 `config.json`：Seattle、San Francisco、Phoenix、Chicago、Houston、Miami、Boston；城市名单仍可调整。
- 保留两套候选方案：次日均温／日较差回归，以及次日天气类别分类；主任务暂不确定。
- 线性模型、Random Forest、SVM 和 persistence / climatology baselines 是候选方案，最终模型和评价设计仍需讨论。
- 拟用 Open-Meteo 提供的 ERA5 历史再分析数据。机场坐标用于选取网格，数据不是机场站点实测；当前配置采用 UTC 日。

**TODO：**与 TA 确认研究问题、数据来源、城市范围、目标变量、分类阈值、模型选择及时间验证方案，再开始实际实验。现有日期、阈值和参数仅为可修改的初始配置，不代表已完成的方法论或实验结论。

## 安装与入口

在本项目目录中使用已有课程环境：

```bash
conda activate data1030
python -m pip install -e . --no-deps
python -m weather_project
```

最后一条命令默认执行 `--stage plan`：**只显示计划，不联网、不训练、不绘图、不创建数据或结果输出。**直接调用 `run_pipeline()` 也采用这一默认行为。

Notebook 是简洁的项目入口，提供配置说明及未来各阶段的注释代码；没有自动下载、训练或缺失文件时自动重建的逻辑。可先阅读，不需要运行它。

需要独立环境时可使用 `conda env create -f environment.yml`，随后激活 `weather1030` 并安装本项目。`requirements.txt` 固定七个核心依赖版本；环境文件另含未固定版本的 JupyterLab、ipykernel、nbformat、nbclient。

## 后续分阶段执行

`--stage` 显式选择操作，各阶段只执行自己负责的工作；缺少前置文件会提示补充，不会自动执行前面的阶段。

| 阶段 | 未来明确选择该阶段时的行为 |
| --- | --- |
| `plan`（默认） | 显示配置和执行计划，无网络、训练、绘图或输出创建 |
| `collect` | 请求 API，缓存响应，保存 daily 数据、质量检查和 collection metadata |
| `prepare` | 只读取本地 daily 数据及 collection metadata，生成 supervised 数据、字典、划分与 preparation metadata |
| `train` | 只读取已准备的数据，训练模型并保存指标与模型 |
| `plot` | 只读取已保存的 daily 数据、指标等文件生成图 |
| `all` | 将全部阶段串联执行，须将来明确选择 |

以下命令仅作为**未来手动执行**的参考，当前均保持注释：

```bash
# python -m weather_project --stage collect --mode full
# python -m weather_project --stage prepare --mode full
# python -m weather_project --stage train --mode full --include-svm --include-range
# python -m weather_project --stage plot --mode full
# python -m weather_project --stage all --mode full --include-svm --include-range
```

`config.json` 管理城市坐标、数据源、日期范围和类别阈值。`--mode smoke` / `full` 选择不同初始时间范围；`--cities boston seattle` 可缩小城市范围。模型、目标和时间切分均应在讨论后调整。

## 目录与版本管理

| 位置 | 用途 |
| --- | --- |
| `Weather_Project.ipynb` | 未执行的项目入口与阶段说明 |
| `src/weather_project/` | 数据、预处理、建模、绘图和阶段调度模块 |
| `config.json` | 可修改的初始实验配置 |
| `data/` | 未来下载的源数据和处理数据 |
| `results/` | 未来指标、元数据与拟合模型 |
| `figures/` | 未来生成的图 |
| `report/` | 未来报告工作 |

仓库保留框架、配置和空目录占位文件。生成的数据 CSV / JSON、图 PNG、结果 CSV / JSON 和模型文件由 `.gitignore` 排除。当前框架不包含已运行的分析结果。

## 上传 GitHub

解压项目包后，将 `DATA1030-weather-project/` 目录中的文件上传到仓库根目录。保留 `src/`、`tests/` 和各目录下的 `.gitkeep`，以及根目录的 `.gitignore`。Notebook 已清空输出，上传后可直接查看框架与流程说明。

## 来源与授权

拟用数据：**Weather data by [Open-Meteo.com](https://open-meteo.com/)**，来源为 Copernicus Climate Change Service / ECMWF ERA5。API 数据采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；将来分享数据或图表时须保留来源、授权链接并说明处理修改。项目原创代码使用 [MIT License](LICENSE)，不改变第三方数据授权。

- [Open-Meteo Historical Weather API 文档](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo 授权与引用](https://open-meteo.com/en/licence)
- Zippenfenig, P. (2023). *Open-Meteo.com Weather API*. [DOI](https://doi.org/10.5281/zenodo.7970649)
- Hersbach et al. (2023). *ERA5 hourly data on single levels from 1940 to present*. ECMWF. [DOI](https://doi.org/10.24381/cds.adbb2d47)

This repository contains a project framework with an output-free notebook. The default command only shows a plan; collection, preparation, training, plotting, and the combined workflow require explicit stage selection.
