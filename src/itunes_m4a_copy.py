import os
import sys
import yaml
import shutil
import logging
import magic
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import multiprocessing
import locale
import unicodedata
import subprocess

# システムのデフォルトエンコーディングを取得
SYSTEM_ENCODING = locale.getpreferredencoding()

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    # デフォルト値の定義
    DEFAULT_CONFIG = {
        'input_dir': '',
        'output_dir': '',
        'ffmpeg_path': 'ffmpeg.exe',
        'aac_bitrate': 320,
        'max_workers': 0,
        'exclude': {
            'directories': [],
            'extensions': []
        }
    }

    input_dir: str
    output_dir: str
    ffmpeg_path: str
    aac_bitrate: int
    max_workers: int
    exclude_dirs: List[str]
    exclude_extensions: List[str]

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """設定ファイルを読み込む"""
        try:
            # 設定ファイルが存在しない場合は、デフォルト値で新規作成
            if not os.path.exists(config_path):
                logger.info("設定ファイルが存在しないため、新規作成します")
                config = cls._create_from_dict(cls.DEFAULT_CONFIG)
                config.save(config_path)
                return config

            # 設定ファイルを読み込み、デフォルト値とマージ
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            # デフォルト値とマージ
            merged_config = cls.DEFAULT_CONFIG.copy()
            merged_config.update(config_data)
            
            # 除外設定のマージ
            exclude = merged_config.get('exclude', {})
            if 'exclude' in config_data:
                merged_config['exclude'] = cls.DEFAULT_CONFIG['exclude'].copy()
                merged_config['exclude'].update(config_data['exclude'])

            return cls._create_from_dict(merged_config)

        except Exception as e:
            logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
            sys.exit(1)

    @classmethod
    def _create_from_dict(cls, config_data: dict) -> 'Config':
        """辞書から設定オブジェクトを作成する"""
        exclude = config_data.get('exclude', {})
        return cls(
            input_dir=config_data.get('input_dir', ''),
            output_dir=config_data.get('output_dir', ''),
            ffmpeg_path=config_data.get('ffmpeg_path', 'ffmpeg.exe'),
            aac_bitrate=config_data.get('aac_bitrate', 320),
            max_workers=config_data.get('max_workers', 0),
            exclude_dirs=exclude.get('directories', []),
            exclude_extensions=exclude.get('extensions', [])
        )

    def save(self, config_path: str) -> None:
        """設定ファイルを保存する"""
        try:
            config_data = {
                'input_dir': self.input_dir,
                'output_dir': self.output_dir,
                'ffmpeg_path': self.ffmpeg_path,
                'aac_bitrate': self.aac_bitrate,
                'max_workers': self.max_workers,
                'exclude': {
                    'directories': self.exclude_dirs,
                    'extensions': self.exclude_extensions
                }
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, allow_unicode=True)
        except Exception as e:
            logger.error(f"設定ファイルの保存に失敗しました: {e}")
            sys.exit(1)

class AudioFileProcessor:
    # iTunesで読み込める拡張子
    ITUNES_EXTENSIONS = {'.m4a', '.mp3', '.aiff', '.aif', '.wav', '.aa', '.aax'}
    # Windowsのファイルシステムエンコーディング
    WINDOWS_ENCODING = 'cp932'

    def __init__(self, config: Config):
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)
        self.ffmpeg_path = config.ffmpeg_path
        # 除外設定を小文字に統一
        self.exclude_dirs = {d.lower() for d in config.exclude_dirs}
        self.exclude_extensions = {e.lower() for e in config.exclude_extensions}

    def _clean_path(self, path: str) -> str:
        """パスの文字列をクリーンアップする"""
        # 前後の空白とクォーテーションを削除
        path = path.strip().strip('"\'')
        # バックスラッシュをスラッシュに統一
        path = path.replace('\\', '/')
        return path

    def _validate_config(self) -> None:
        """設定値の検証を行う"""
        if not os.path.exists(self.config.ffmpeg_path):
            path = self._clean_path(input("ffmpegのパスを入力してください: "))
            if not os.path.exists(path):
                logger.error("無効なffmpegのパスです")
                sys.exit(1)
            self.config.ffmpeg_path = path

        if not self.config.input_dir or not os.path.exists(self.config.input_dir):
            path = self._clean_path(input("入力フォルダのパスを入力してください: "))
            if not os.path.exists(path):
                logger.error("無効な入力フォルダパスです")
                sys.exit(1)
            self.config.input_dir = path
            self.input_dir = Path(path)

        if not self.config.output_dir:
            path = self._clean_path(input("出力フォルダのパスを入力してください: "))
            self.config.output_dir = path
            self.output_dir = Path(path)

        os.makedirs(self.config.output_dir, exist_ok=True)

    def _encode_path(self, path: str) -> bytes:
        """パスをバイト列にエンコードする"""
        try:
            return path.encode(self.WINDOWS_ENCODING)
        except UnicodeEncodeError:
            # cp932で変換できない文字がある場合はutf-8を試す
            return path.encode('utf-8')

    def _decode_path(self, path: Union[str, bytes]) -> str:
        """パスを文字列にデコードする"""
        if isinstance(path, bytes):
            try:
                return path.decode(self.WINDOWS_ENCODING)
            except UnicodeDecodeError:
                # cp932で変換できない場合はutf-8を試す
                return path.decode('utf-8', errors='replace')
        return path

    def _normalize_string(self, s: str) -> str:
        """文字列を正規化する"""
        return unicodedata.normalize('NFKC', s)

    def _should_exclude(self, path: Path, is_input: bool = False) -> bool:
        """パスが除外対象かどうかを判定する"""
        # 出力ディレクトリの場合は除外しない
        if not is_input:
            return False
            
        # ディレクトリの除外チェック
        for part in path.parts:
            if part.lower() in self.exclude_dirs:
                return True
        
        # 拡張子の除外チェック
        if path.suffix.lower() in self.exclude_extensions:
            return True
            
        return False

    def scan_directory(self, directory: Path, total_percent: float, pbar: tqdm = None) -> Dict[str, datetime]:
        """ディレクトリ内の音声ファイルをスキャンする"""
        files = {}
        try:
            # 入力ディレクトリかどうかを判定
            is_input = directory == self.input_dir
            
            # 最初にファイル数をカウント（進捗計算用）
            total_files = sum(1 for _ in directory.rglob('*') if _.is_file())
            if total_files > 0:
                step = total_percent / total_files
            else:
                step = 0
            
            for file_path in directory.rglob('*'):
                try:
                    if file_path.is_file() and not self._should_exclude(file_path, is_input):
                        if pbar:
                            pbar.update(step)
                        try:
                            # パスをデコード
                            rel_path = self._decode_path(str(file_path.relative_to(directory)))
                            abs_path = self._decode_path(str(file_path))

                            # iTunesで読み込める拡張子は除外（入力ディレクトリの場合のみ）
                            if not is_input or file_path.suffix.lower() not in self.ITUNES_EXTENSIONS:
                                try:
                                    with open(file_path, 'rb') as f:
                                        data = f.read(4096)
                                        mime = magic.from_buffer(data, mime=True)
                                        if mime and mime.startswith('audio/'):
                                            files[rel_path] = datetime.fromtimestamp(os.path.getmtime(file_path))
                                except Exception as e:
                                    logger.debug(f"ファイルの種類の判定に失敗しました: {abs_path} - {e}")
                        except Exception as e:
                            logger.debug(f"相対パスの取得に失敗しました: {file_path} - {e}")
                except Exception as e:
                    logger.debug(f"ファイル情報の取得に失敗しました: {file_path} - {e}")
        except Exception as e:
            logger.debug(f"ディレクトリのスキャンに失敗しました: {directory} - {e}")
        return files

    def get_output_path(self, input_path: str) -> str:
        """出力ファイルパスを取得する"""
        path = Path(input_path)
        # 出力は全てm4aに変換
        return str(path.with_suffix('.m4a'))

    def get_input_path(self, output_path: str) -> str:
        """出力ファイルパスから入力ファイルパスを取得する"""
        try:
            path = Path(output_path)
            # 入力ファイルを探すために、元のファイル名（拡張子を除く）を使用
            base_name = path.stem
            parent = path.parent
            # 入力ディレクトリ内の同じ相対パスを探す
            input_dir_path = self.input_dir / parent
            if input_dir_path.exists():
                # 同じベース名を持つファイルを探す（特殊文字を考慮）
                for file in input_dir_path.iterdir():
                    try:
                        if file.is_file():
                            # ファイル名の比較は正規化して行う
                            file_stem = self._decode_path(file.stem)
                            if self._normalize_string(file_stem) == self._normalize_string(base_name):
                                # 除外する拡張子はスキップ
                                if file.suffix.lower() in self.exclude_extensions:
                                    continue
                                # iTunesで読み込めない拡張子を持つファイルを見つけた
                                if file.suffix.lower() not in self.ITUNES_EXTENSIONS:
                                    rel_path = str(parent / file.name) if parent != Path('.') else file.name
                                    return self._decode_path(rel_path)
                    except Exception as e:
                        logger.debug(f"ファイル名の比較に失敗しました: {file} - {e}")
            return output_path
        except Exception as e:
            logger.debug(f"入力パスの取得に失敗しました: {output_path} - {e}")
            return output_path

    def count_target_files(self, directory: Path, is_input: bool = False) -> int:
        """処理対象となるファイル数をカウントする"""
        count = 0
        try:
            for file_path in directory.rglob('*'):
                try:
                    if file_path.is_file() and not self._should_exclude(file_path, is_input):
                        # 入力ディレクトリの場合はiTunesで読み込める拡張子を除外
                        if not is_input or file_path.suffix.lower() not in self.ITUNES_EXTENSIONS:
                            try:
                                with open(file_path, 'rb') as f:
                                    data = f.read(4096)
                                    mime = magic.from_buffer(data, mime=True)
                                    if mime and mime.startswith('audio/'):
                                        count += 1
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        return count

    def compare_files(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """入力と出力のファイルを比較する"""
        # 進捗表示の準備（100%を3つのフェーズに分割）
        with tqdm(total=100, desc="ファイル分析中", bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]') as pbar:
            input_files = self.scan_directory(self.input_dir, 30, pbar)
            output_files = self.scan_directory(self.output_dir, 30, pbar)

            to_copy = []  # 新規コピー
            to_update = []  # 更新
            to_delete = []  # 削除
            unchanged = []  # 変更なし
            
            # 比較処理の進捗計算用
            total_comparisons = len(input_files) + len(output_files)
            if total_comparisons > 0:
                comparison_step = 40 / total_comparisons
            else:
                comparison_step = 0
            
            # 入力ファイルの処理
            for in_path, in_time in input_files.items():
                out_path = self.get_output_path(in_path)
                
                if out_path not in output_files:
                    to_copy.append((in_path, out_path))
                else:
                    out_time = output_files[out_path]
                    if in_time > out_time:
                        to_update.append((in_path, out_path))
                    else:
                        unchanged.append(in_path)
                pbar.update(comparison_step)

            # 出力ファイルの処理
            for out_path in output_files:
                in_path = self.get_input_path(out_path)
                if in_path not in input_files:
                    to_delete.append((in_path, out_path))
                pbar.update(comparison_step)

        # 分析完了後にログを出力
        for in_path, out_path in to_delete:
            logger.info(f"削除: {out_path} (元: {in_path})")
                    
        # 戻り値用にパスのみのリストに変換
        return (
            [p[0] for p in to_copy],
            [p[0] for p in to_update],
            [p[1] for p in to_delete],
            unchanged
        )

    def check_disk_space(self, files_to_process: List[str]) -> bool:
        """必要な容量をチェックする"""
        required_space = 0
        for file_path in files_to_process:
            try:
                size = os.path.getsize(self.input_dir / file_path)
                # 安全のため、元のサイズの1.5倍を見積もる
                required_space += size * 1.5
            except Exception as e:
                logger.warning(f"ファイルサイズの取得に失敗しました: {file_path} - {e}")

        free_space = shutil.disk_usage(self.output_dir).free
        if required_space > free_space:
            logger.error(
                f"十分なディスク容量がありません。必要: {required_space / 1024 / 1024:.1f}MB, "
                f"利用可能: {free_space / 1024 / 1024:.1f}MB"
            )
            return False
        return True

    def _build_ffmpeg_command(self, input_path: str, input_abs_path: str, output_abs_path: str) -> list:
        """ffmpegコマンドを構築する"""
        # 入力ファイルの形式を判定
        is_lossless = any(fmt in input_path.lower() for fmt in ['.flac', '.wav', '.aiff', '.alac'])
        is_ogg = input_path.lower().endswith('.ogg')

        # 基本オプション
        cmd = [
            str(self.ffmpeg_path),
            '-hide_banner',     # バナー表示を抑制
            '-loglevel', 'warning',  # warning以上のみ表示
            '-i', input_abs_path,    # 入力ファイル
        ]

        # メタデータオプション
        if is_ogg:
            cmd.extend(['-map_metadata', '0:s'])

        # エンコードオプション
        cmd.extend([
            '-c:v', 'copy',     # ビデオストリームはそのままコピー
            '-c:a', 'alac' if is_lossless else 'aac',  # オーディオコーデック
        ])

        # ビットレートオプション（非可逆圧縮の場合のみ）
        if not is_lossless:
            cmd.extend(['-b:a', f'{self.config.aac_bitrate}k'])

        # 出力オプション
        cmd.extend(['-y', output_abs_path])  # 出力先（上書き許可）

        return cmd

    def process_file(self, input_path: str) -> bool:
        """ファイルを処理する"""
        try:
            input_abs_path = str(self.input_dir / input_path)
            output_abs_path = str(self.output_dir / self.get_output_path(input_path))

            # 出力先のディレクトリを作成
            os.makedirs(os.path.dirname(output_abs_path), exist_ok=True)

            # ffmpegコマンドを構築
            cmd = self._build_ffmpeg_command(input_path, input_abs_path, output_abs_path)

            # ffmpegを実行
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                if result.stderr and b'error' in result.stderr.lower():  # エラーメッセージのみログ出力
                    logger.debug(f"ffmpegエラー: {result.stderr.decode(SYSTEM_ENCODING, errors='replace')}")
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpegの実行に失敗しました: {e.stderr.decode(SYSTEM_ENCODING, errors='replace')}")
                return False

            return True
        except Exception as e:
            logger.error(f"ファイルの処理に失敗しました: {input_path} - {e}")
            return False

    def process_files(self, files: List[str]) -> None:
        """ファイルを並列処理する"""
        # 並列処理数が0の場合は自動判断
        workers = self.config.max_workers
        if workers <= 0:
            # CPU数の75%を使用（最低1）
            workers = max(1, int(multiprocessing.cpu_count() * 0.75))
            logger.info(f"並列処理数を自動設定しました: {workers}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(
                executor.map(self.process_file, files),
                total=len(files),
                desc="ファイル処理中"
            ))

    def delete_files(self, files: List[str]) -> None:
        """ファイルを削除する"""
        for file_path in tqdm(files, desc="不要なファイルを削除中"):
            try:
                os.remove(self.output_dir / file_path)
            except Exception as e:
                logger.error(f"ファイルの削除に失敗しました: {file_path} - {e}")

    def run(self) -> None:
        """メイン処理を実行する"""
        logger.info(f"入力フォルダ: {self.config.input_dir}")
        logger.info(f"出力フォルダ: {self.config.output_dir}")

        # デバッグログを有効化
        logger.setLevel(logging.DEBUG)

        # 設定値の検証
        self._validate_config()

        # ファイルの比較
        to_copy, to_update, to_delete, unchanged = self.compare_files()

        # 確認
        logger.info(f"処理結果: 新規={len(to_copy)}, 更新={len(to_update)}, 削除={len(to_delete)}, 維持={len(unchanged)}")

        if not (to_copy or to_update or to_delete):
            logger.info("処理するファイルがありません")
            return

        # ディスク容量チェック
        if not self.check_disk_space(to_copy + to_update):
            return

        # 実行確認
        if input("処理を開始しますか？ (y/n): ").lower() != 'y':
            logger.info("処理を中止しました")
            return

        # ファイルの処理
        if to_copy or to_update:
            self.process_files(to_copy + to_update)

        # 不要なファイルの削除
        if to_delete:
            self.delete_files(to_delete)

        logger.info("処理が完了しました")

def main():
    try:
        config = Config.load('config.yaml')
        processor = AudioFileProcessor(config)
        processor.run()
        # 更新された設定を保存
        config.save('config.yaml')
    except KeyboardInterrupt:
        logger.info("\n処理を中断しました")
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")
    finally:
        # プログラム終了前にユーザーの入力を待つ
        input("\nEnterキーを押して終了してください...")
        sys.exit(1)

if __name__ == '__main__':
    main()
