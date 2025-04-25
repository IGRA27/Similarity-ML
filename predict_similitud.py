import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_synonyms(excel_path: str):
    """
    Carga el Excel de sinónimos y devuelve:
      - df_syn: DataFrame original
      - synonyms: lista de listas de sinónimos por fila
    """
    df = pd.read_excel(excel_path)
    # Asume que las columnas de opciones contienen 'Opcion' en su nombre
    opt_cols = [c for c in df.columns if 'Opcion' in c]
    synonyms = []
    for _, row in df.iterrows():
        opts = [str(row[c]).strip()
                for c in opt_cols
                if pd.notna(row[c]) and str(row[c]).strip()]
        synonyms.append(opts)
    return df, synonyms


def embed_texts(model, texts: list, batch_size: int = 32):
    """
    Obtiene embeddings de una lista de textos con SBERT.
    """
    return model.encode(texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True)


def main():
    parser = argparse.ArgumentParser(
        description='Calcula similitud entre Asuntos y Sinónimos con SBERT.')
    parser.add_argument('synonyms_excel', type=str,
                        help='Ruta al Excel de sinónimos (DataFrame 1).')
    parser.add_argument('asuntos_excel', type=str,
                        help='Ruta al Excel de asuntos (DataFrame 2).')
    parser.add_argument('--output', type=str, default='resultado_similitud.xlsx',
                        help='Ruta de salida para el Excel de resultados.')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2',
                        help='Nombre del modelo SBERT para embeddings.')
    args = parser.parse_args()

    # 1) Cargo los datos
    df_syn, synonyms = load_synonyms(args.synonyms_excel)
    df_asuntos = pd.read_excel(args.asuntos_excel)

    # 2) Preparo textos de sinónimos (uno por fila) concatenando con separador
    syn_texts = [' | '.join(opts) for opts in synonyms]

    # 3) Cargo SBERT y obtengo embeddings
    model = SentenceTransformer(args.model_name)
    syn_emb = embed_texts(model, syn_texts)
    asuntos_list = df_asuntos['Asunto'].astype(str).tolist()
    asuntos_emb = embed_texts(model, asuntos_list)

    # 4) Similitud coseno y recogida del máximo por fila
    sim_matrix = cosine_similarity(asuntos_emb, syn_emb)
    max_sims = sim_matrix.max(axis=1)  # valor de similitud [0..1]

    # 5) Preparo DataFrame de salida
    df_out = pd.DataFrame({
        'Asunto': asuntos_list,
        **({'ID': df_asuntos['ID']} if 'ID' in df_asuntos else {}),
        **({'Correo': df_asuntos['Correo']} if 'Correo' in df_asuntos else {}),
        'Similitud': (max_sims * 100).round(2).astype(str) + '%'
    })

    # 6) Exporto a Excel
    df_out.to_excel(args.output, index=False)
    print(f"[+] Resultados guardados en: {args.output}")


if __name__ == '__main__':
    main()
