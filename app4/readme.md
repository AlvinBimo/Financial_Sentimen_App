# Linux Problem 
1. dont forget to change the folder owner to user not root, this problem can lead to permission error when tried to run the app

# Major Update
1. **Versi pada Docker ini menggunakan torch yang tidak memerlukan ROCm**
2. **Mengtranslasi seluruh teks** pada aplikasi menjadi bahasa Indonesia &#8594; affected in app.py
3. **Mengnonaktifkan sumber berita** **detik.com** dikarenakan terdapat permasalahan saat melakukan scraping &#8594; affected in config.py and app.py and deleted detik.py from previous version

4. **Perbedaan tampilan pada dua tombol utama**
    Memisahkan gaya visual tombol "Kumpulkan Data & Analisa Data" dan "Load Previous Results" agar lebih jelas dan mudah dibedakan.

5. **Perubahan warna input filter**
    Mengganti warna isian filter dari merah ke warna lain (selain merah, hijau, abu, kuning) untuk mengurangi misleading warning effect.

6. **Penerapan container pada seluruh visualisasi**
    Seluruh elemen grafik dan visual output kini ditempatkan dalam container masing-masing untuk tampilan lebih rapi dan terstruktur.

7. **Penambahan hyperlink Pada Seluruh Hasil Scraping**
    Menampilkan tautan sumber Putusan MA langsung dari hasil pencarian.

8. **Penyesuaian validasi mandatory feedback**
    Jika dataframe kosong → feedback tidak wajib

    Jika dataframe < 5 → feedback wajib 1

    Jika dataframe ≥ 5 → feedback wajib 5
    Tombol analisis tetap aktif sesuai kondisi di atas.

9. **Penambahan logo/identitas perusahaan**
    Identitas perusahaan ditampilkan dalam aplikasi untuk kebutuhan branding.

10. **Penataan axis pada grafik**
    Penulisan axis distandarkan: kapitalisasi di awal kata dan konsisten dengan nama news source.

11. **Pemindahan pesan wajib feedback ke dalam container hasil**
    Wording “Please provide feedback…” kini ditampilkan di dalam container Hasil Analisis Sentimen Berserta Artikel.

12. **Membedakan container Putusan MA dan container berita**
    Output Putusan MA dipisahkan dari artikel berita untuk navigasi yang lebih jelas.

13. **Peraturan feedback duplikat**
    Jika pengguna memberi feedback lebih dari sekali pada satu statement:

    Feedback lama direplace

    Progress tidak bertambah
    Progress hanya bertambah jika feedback diberikan pada statement baru.


# Minor Update
1. Mengurangi nilai default max article dari tiap sumber berita menjadi 100 &#8594; affected in app.py 
2. Delete model to only use 1 model -> affected in config.py and app.py
3. Add help description on the start and date &#8594; affected in app.py
4. Limit the Start date to only 1 Januari 2020 and the end date will be limited to user "current date" &#8594; affected in app.py
5. Add timer for scraping and inference &#8594; app.py and run_analyze.py (for the news source)
6. Increase Model Batch Size for inference to 128 &#8594; config.py
7. Add streamlit marquee
8. Mengisi nilai default dari periode awal menjadi 1 Januari 2020 dan akhir periode menjadi tanggal "hari ini"
9. Menambahkan feedback counter maksimal 5

10. Memperpendek waktu jeda running text pada marquee 

11. Penyesuaian ukuran kolom pada output Excel 

12. Penebalan teks “Ditemukan” dan “Tidak Ditemukan” pada hasil Daftar Hitam INAPROC dan Putusan MA

13. Penambahan disclaimer pada hasil Putusan MA
“Data yang ditampilkan merujuk pada pencarian per kata…”

14. Pengaturan format nama file
Contoh: "inaproc - (nama keyword)".

15. Memperbesar font pada diagram 

16. Menghapus fitur Zoom pada grafik Sentiment Distribution by News Source.

17. Checklist “Highlight Relevant Sentence” dibuat default aktif.

18. Hyperlink berita dibuka pada tab baru untuk kenyamanan pengguna.

Catatan internal:
Percobaan membuat container geser-geser sentiment over time → tidak dapat dilakukan karena keterbatasan Streamlit.