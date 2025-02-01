# iTunesM4ACopy

iTunesで読み込めない音声ファイル(FLAC等)を抽出し、M4A（ALAC/AAC）に変換して別フォルダにコピーするツール。
自分用の雑クオリティなので正しい動作は保証しない。
少なくとも入力フォルダが破壊されることはないはず。
Claude 3.5 Sonnetを使用して開発。

## 必須
- ffmpeg.exe

## 特徴

- iTunesで読み込めない音声ファイルを自動検出
- FLACとoggへの対応を確認済（他の拡張子は未確認）
- 可逆圧縮（FLAC等）→ALAC、非可逆圧縮→AACに自動変換
- アートワークを含むメタデータを維持
- 並列処理による高速な変換
- 必要なディスク容量を事前チェック

## iTunesに読み込むまでの流れ

1. 音声ファイルをまとめたフォルダAと、M4A変換先のフォルダBを用意
1. 当プログラムで変換処理を行う
2. AとBをiTunesにD&D

## 設定

`config.yaml`ファイルで以下の設定が可能です：

- `ffmpeg_path`: ffmpegのパス
- `input_dir`: 入力フォルダのパス
- `output_dir`: 出力フォルダのパス
- `aac_bitrate`: AAC変換時のビットレート（デフォルト: 320kbps）
- `max_workers`: 並列処理数（デフォルト: 0、自動で最適な値を設定）
- `exclude`: 除外設定
  - `directories`: 除外するフォルダ名のリスト
  - `extensions`: 除外するファイル拡張子のリスト
