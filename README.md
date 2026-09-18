# 谱境研音 / Soundscape Analyse

一个基于 Python 与 Streamlit 的音频、频谱和乐谱辅助分析工具。项目提供浏览器工作台、命令行入口、交互式图表与结构化导出。

## 功能概览

### 音频工作台

- 支持 `WAV / FLAC / AIFF / OGG / MP3 / M4A`
- 显示同步波形与高对比频谱图
- 支持线性或对数频率刻度
- 提取 RMS、频谱质心、带宽、rolloff、flatness、flux 与 onset strength 等特征
- 自动生成候选事件点与边界，并支持人工调整时间和标签
- 提供局部试听、局部波形和局部频谱
- 导出 `CSV / JSON / PNG / TXT`

### 乐谱工作台

- 支持 `MusicXML / MXL / MIDI / KRN`
- 提供音高、音级、音程与和声信息浏览
- 支持候选结果的人工检查与编辑
- 可从内置的公开乐谱目录载入示例文件

## 在线部署

本项目可部署到 [Streamlit Community Cloud](https://share.streamlit.io/)：

1. 在 Streamlit Community Cloud 中连接 GitHub。
2. 选择仓库 `LiZhuoming-lab/soundscape-analyse`。
3. 分支选择 `main`。
4. 主文件路径填写 `app.py`。
5. 建议使用 Python 3.12。
6. 部署后，`main` 分支更新会自动触发重新部署。

仓库已经包含：

- `requirements.txt`：Python 依赖
- `packages.txt`：系统依赖
- `.streamlit/config.toml`：Streamlit 配置

云端实例资源有限。较长或多声道文件建议在本地运行，分析完成后及时下载导出结果。

## 本地运行

需要 Python 3.11 或 Python 3.12。

```bash
git clone https://github.com/LiZhuoming-lab/soundscape-analyse.git
cd soundscape-analyse
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Windows PowerShell：

```powershell
git clone https://github.com/LiZhuoming-lab/soundscape-analyse.git
cd soundscape-analyse
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

启动后访问：

```text
http://localhost:8501
```

## 命令行

```bash
python3 -m spectral_tool.cli your_audio.wav
```

可选参数示例：

```bash
python3 -m spectral_tool.cli your_audio.wav \
  --channel mix \
  --target-sr 32000 \
  --n-fft 4096 \
  --hop-length 1024 \
  --min-event-distance 5 \
  --threshold-sigma 1.0 \
  --spectrogram-dynamic-range 90
```

## 导出内容

- `events.csv`
- `sections.csv`
- `feature_table.csv`
- `analysis.json`
- `summary.txt`
- 波形、频谱、新颖度与事件密度 PNG

自动结果用于辅助定位和整理，建议结合原始音频或乐谱进行人工复核。

## 项目结构

```text
app.py                         Streamlit 入口
spectral_tool/analysis.py      音频特征与事件检测
spectral_tool/visualization.py 图表与交互可视化
spectral_tool/symbolic_analysis.py 乐谱解析
spectral_tool/ui/              Streamlit 工作台
tests/                         自动化测试
```

## 测试

```bash
python3 -m pytest -q
```

## 隐私

上传到 Streamlit 的文件仅用于当前会话处理。应用不会主动把用户上传的音频、乐谱或标注写入本仓库。使用公共部署处理敏感材料前，请自行确认平台的数据与访问政策。

## 开源依赖

本项目使用 [music21](https://github.com/cuthbertLab/music21) 及其他 `requirements.txt` 中列出的开源库。

## License

[MIT License](LICENSE)
