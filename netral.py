import pandas as pd
import os
from datetime import datetime

# ==============================
# 🔧 KONFIGURASI DASAR
# ==============================
EXCEL_FILE = "data/HIV_estimates_from_1990-to-present.xlsx"
EXPORT_DIR = "data/csv_exports"
FINAL_OUTPUT = "data/preprocessed.csv"


def export_all_sheets(excel_path=EXCEL_FILE, output_dir=EXPORT_DIR):
    """
    Ekspor semua sheet dari file Excel ke CSV terpisah tanpa filter.
    Hanya menghapus baris & kolom kosong penuh.
    """
    if not os.path.exists(excel_path):
        print(f"❌ File tidak ditemukan: {excel_path}")
        return []

    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Membaca semua sheet dari: {excel_path}")
    sheets = pd.read_excel(excel_path, sheet_name=None, header=None)
    print(f"📚 Ditemukan {len(sheets)} sheet dalam file Excel ini.\n")

    exported_files = []
    for sheet_name, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")

        safe_name = sheet_name.replace("/", "_").replace("\\", "_")
        output_path = os.path.join(output_dir, f"{safe_name}.csv")

        df.to_csv(output_path, index=False, header=False, encoding="utf-8-sig")
        exported_files.append(output_path)

        print(f"✅ Sheet '{sheet_name}' berhasil disimpan ke: {output_path}")

    print("\n🎉 Semua sheet telah berhasil diekspor ke CSV!")
    return exported_files


def combine_csv_exports(output_dir=EXPORT_DIR, output_file=FINAL_OUTPUT):
    """
    Gabungkan semua CSV hasil ekspor menjadi satu file preprocessed.csv.
    Setiap baris diberi label asal sheet agar mudah dilacak.
    """
    if not os.path.exists(output_dir):
        print(f"⚠️ Folder {output_dir} belum ada. Jalankan ekspor dulu.")
        return None

    all_dfs = []
    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

    if not csv_files:
        print("⚠️ Tidak ada file CSV ditemukan untuk digabungkan.")
        return None

    print(f"\n🔗 Menggabungkan {len(csv_files)} file CSV...")

    for file in csv_files:
        path = os.path.join(output_dir, file)
        try:
            df = pd.read_csv(path, header=None)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df.insert(0, "source_sheet", file)
            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Gagal membaca {file}: {e}")

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.insert(0, "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"✅ Dataset gabungan berhasil dibuat: {output_file}")
    print(f"📈 Total baris: {len(merged)}")
    return output_file


def prepare_dataset_pipeline():
    """
    Jalankan seluruh pipeline preprocessing:
    1. Ekspor semua sheet → CSV
    2. Gabungkan semua CSV jadi preprocessed.csv
    """
    print("🚀 Menjalankan pipeline konversi & persiapan dataset...\n")
    exported_files = export_all_sheets()
    if exported_files:
        final = combine_csv_exports()
        if final:
            print("\n🎯 Proses selesai! Dataset siap digunakan untuk simulasi.")
        else:
            print("⚠️ Penggabungan CSV gagal.")
    else:
        print("⚠️ Tidak ada sheet berhasil diekspor.")


# ==============================
# ▶️ MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    prepare_dataset_pipeline()
