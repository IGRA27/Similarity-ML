#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    p = argparse.ArgumentParser(
        description='Top-3 similitud entre asuntos y catálogo de sinónimos')
    p.add_argument('artefacto',
                   help='artefacto_similitud.joblib')
    p.add_argument('asuntos_excel',
                   help='Excel diario con columnas Asunto, ID, Correo')
    p.add_argument('--output', '-o', default='resultados_top3.xlsx',
                   help='Excel de salida con top-3')
    args = p.parse_args()

    # 1) Carga artefacto entrenado
    arte = joblib.load(args.artefacto)
    model_name = arte['model_name']
    syn_texts = arte['syn_texts']
    syn_emb   = arte['syn_emb']

    # 2) Carga asuntos nuevos
    df_as = pd.read_excel(args.asuntos_excel)
    texts = df_as['Asunto'].astype(str).tolist()

    # 3) Embeddings de asuntos
    model = SentenceTransformer(model_name)
    as_emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # 4) Matriz de similitud y top-3
    sim_mat = cosine_similarity(as_emb, syn_emb)
    # Para cada row, extraemos índices de top-3 similares
    topk = 3
    idx_sorted = sim_mat.argsort(axis=1)[:, ::-1][:, :topk]
    vals_sorted = -(-sim_mat).argsort(axis=1)[:, :topk]  # pero mejor obtener valores directos:
    top_vals = np.take_along_axis(sim_mat, idx_sorted, axis=1)

    # Formateamos en porcentaje
    top_pcts = [[f"{v*100:.2f}%" for v in row] for row in top_vals]

    # 5) Construye DataFrame de salida
    out = pd.DataFrame({
        'Asunto': texts,
        **({'ID': df_as['ID']} if 'ID' in df_as else {}),
        **({'Correo': df_as['Correo']} if 'Correo' in df_as else {})
    })

    # Añadimos columnas top-3
    for rank in range(topk):
        out[f'Match_{rank+1}'] = [syn_texts[i] for i in idx_sorted[:, rank]]
        out[f'Sim_{rank+1}']   = [top_pcts[r][rank] for r in range(len(texts))]

    # 6) Exporta
    out.to_excel(args.output, index=False)
    print(f"[+] Resultados top-3 guardados en: {args.output}")

if __name__ == '__main__':
    main()
