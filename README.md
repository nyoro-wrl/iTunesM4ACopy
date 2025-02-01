# iTunesM4ACopy

iTunesで読み込めない音声ファイルを、iTunesで読み込める形式（M4A/ALAC/AAC）に変換してコピーするツール。
Claude 3.5 Sonnetを使用。

## 特徴

- iTunesで読み込めない音声ファイルを自動検出
- flacとoggへの対応を確認済（他の拡張子は未確認）
- 可逆圧縮（FLAC等）→ALAC、非可逆圧縮→AACに自動変換
- アートワークを含むメタデータを維持
- 並列処理による高速な変換
- 必要なディスク容量を事前チェック

## 設定

`config.yaml`ファイルで以下の設定が可能です：

- `input_dir`: 入力フォルダのパス
- `output_dir`: 出力フォルダのパス
- `ffmpeg_path`: ffmpegのパス
- `aac_bitrate`: AAC変換時のビットレート（デフォルト: 320kbps）
- `max_workers`: 並列処理数（デフォルト: 0、自動で最適な値を設定）
- `exclude`: 除外設定
  - `directories`: 除外するディレクトリ名のリスト
  - `extensions`: 除外するファイル拡張子のリスト
