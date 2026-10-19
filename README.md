# UTS Sains Data Genom: kardiomiopati dilatasi (microarray dan RNA-seq)

Analisis diferensial ekspresi gen, enrichment *Gene Ontology*, clustering, biclustering,
dan perbandingan lintas platform pada miokard ventrikel kiri manusia.

| Sumber | Accession | Teknologi | Kelompok yang dianalisis |
|---|---|---|---|
| NCBI GEO | GSE5406 | Microarray Affymetrix HG-U133A (GPL96) | 86 idiopathic DCM vs 16 non-failing |
| NCBI GEO | GSE116250 | RNA-seq Illumina HiSeq 2500 (GPL16791) | 37 DCM vs 14 non-failing |

Kelompok kardiomiopati iskemik pada kedua studi dikeluarkan agar perbandingan
tidak bercampur dengan etiologi iskemik.

## Isi repositori

```
analisis/UTS_Analisis_Genom.py    pipeline lengkap (satu berkas)
analisis/UTS_Analisis_Genom.ipynb  notebook versi sama dengan keluaran tersimpan
data/                              anotasi probe ke simbol gen (GPL96)
```

## Menjalankan

```bash
uv run python analisis/UTS_Analisis_Genom.py
```

Berkas data besar diunduh otomatis dari NCBI GEO bila belum ada. Anotasi platform
GPL96 sudah disertakan sebagai `data/gpl96_probe_symbol.csv`.

Notebook dapat dibangun ulang dari skrip:

```bash
uv run jupytext --to ipynb analisis/UTS_Analisis_Genom.py -o analisis/UTS_Analisis_Genom.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 analisis/UTS_Analisis_Genom.ipynb
```

## Tahap analisis

1. Praproses microarray: baca series matrix, normalisasi kuantil antar-array,
   saring probe, petakan probe ke simbol gen, gabungkan probe per gen.
2. Praproses RNA-seq: RPKM ke skala log, saring gen berdasarkan ekspresi,
   buang nilai penskalaan artefak, gabungkan simbol gen.
3. Uji t termoderasi (empirical Bayes ala limma) dan koreksi Benjamini-Hochberg.
   Gen diferensial bila q < 0,05 dan |log2FC| >= 0,585.
4. Enrichment *Gene Ontology* dan KEGG dengan Enrichr.
5. Anotasi gen penyandi protein memakai mygene.
6. PCA, clustering hierarkis, heatmap, dan biclustering spektral.
7. Perbandingan gen bersama, korelasi Spearman, dan Jaccard index.

## Ringkasan hasil

| Metrik | Microarray | RNA-seq |
|---|---|---|
| Gen diuji | 7.783 | 16.804 |
| Gen diferensial | 163 (2,09 persen) | 1.783 (10,61 persen) |
| Gen naik | 74 (45,4 persen) | 1.124 (63,04 persen) |
| Gen turun | 89 (54,6 persen) | 659 (36,96 persen) |
| ARI klaster hierarkis | 0,093 | 0,460 |
| Silhouette (k=2) | 0,128 | 0,202 |

Gen bersama kedua platform: 6.080, korelasi Spearman 0,253, 63 gen diferensial
bersama, Jaccard 3,35 persen, arah perubahan konsisten 87,3 persen.

## Pembuktian

Kode ditulis dengan bantuan AI (OpenCode) dan diverifikasi dengan menjalankan ulang
seluruh pipeline dari data GEO, memeriksa istilah GO pada Enrichr, serta mencocokkan
rujukan pada PubMed.
