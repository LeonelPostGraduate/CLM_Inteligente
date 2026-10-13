# database.py
import json
import psycopg
from psycopg.types.json import Jsonb
from models import Contrato, Documento, DocumentoEmbedding, AnaliseCognitiva, AnaliseRisco

class DatabaseRepository:
    def __init__(self, connection_string: str):
        self.conn_str = connection_string

    def obter_conexao(self):
        """Retorna uma conexão ativa com o banco PostgreSQL."""
        return psycopg.connect(self.conn_str)

    def salvar_contrato_inicial(self, contrato: Contrato):
        """Salva a estrutura base de um contrato no banco."""
        # Nota: O bloco with no psycopg3 gerencia a transação e commita automaticamente
        with self.obter_conexao() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO contratos (id, numero_contrato, objeto, valor_global, vigencia_inicio, vigencia_fim, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (str(contrato.id), contrato.numero_contrato, contrato.objeto, contrato.valor_global, 
                     contrato.vigencia_inicio, contrato.vigencia_fim, contrato.status.value)
                )

    def salvar_documento_e_embedding(self, contrato_id: str, documento: Documento, embedding: DocumentoEmbedding):
        """Persiste um documento e seu respectivo vetor traduzido para o pgvector."""
        with self.obter_conexao() as conn:
            with conn.cursor() as cur:
                # 1. Insere o Documento
                cur.execute(
                    """
                    INSERT INTO documentos (id, contrato_id, nome_arquivo, tipo_documento, versao, hash_arquivo, gcp_storage_url, conteudo_texto)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (str(documento.id), str(contrato_id), documento.nome_arquivo, documento.tipo_documento, 
                     documento.versao, documento.hash_arquivo, documento.gcp_storage_url, documento.conteudo_texto)
                )
                
                # 2. Formata explicitamente a lista de floats para string de vetor '[v1,v2,...]' exigida pelo pgvector
                vetor_formatado = "[" + ",".join(map(str, embedding.vetor_embedding)) + "]"
                
                # 3. Insere o Vetor Numérico
                cur.execute(
                    """
                    INSERT INTO documento_embeddings (id, documento_id, chunk_texto, vetor_embedding, numero_pagina, modelo_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (str(embedding.id), str(documento.id), embedding.chunk_texto, vetor_formatado, 
                     embedding.numero_pagina, embedding.modelo_embedding)
                )

    
def salvar_analise_e_riscos(self, contrato_id: str, analise: AnaliseCognitiva, riscos: list[AnaliseRisco]):
        """Salva a análise do Gemini (JSONB) e desmembra os riscos nas tabelas relacionais."""
        with self.obter_conexao() as conn:
            with conn.cursor() as cur:
                # 1. Salva a análise cognitiva com payload JSONB nativo
                cur.execute(
                    """
                    INSERT INTO analises_cognitivas (id, contrato_id, tipo_analise, modelo_ia, versao_modelo, score_confianca, resultado_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (str(analise.id), str(contrato_id), analise.tipo_analise.value, analise.modelo_ia, 
                     analise.versao_modelo, float(analise.score_confianca), Jsonb(analise.resultado_json))
                )
                
                # 2. Desmembra e salva os riscos identificados relacionalmente
                for risco in riscos:
                    cur.execute(
                        """
                        INSERT INTO analises_risco (id, contrato_id, nivel_risco, descricao, plano_mitigacao)
                        VALUES (%s, %s, %s, %s, %s);
                        """,
                        (str(risco.id), str(contrato_id), risco.nivel_risco, risco.descricao, risco.plano_mitigacao)
                    )

    def buscar_contexto_semantico(self, vetor_busca: list[float], limite: int = 3) -> list[str]:
        """
        [MÉTODO DE ACESSO - FUNDAMENTAL PARA O RAG]
        Realiza a busca por similaridade de cosseno (operador <=>) no PostgreSQL.
        Compara o vetor da pergunta do usuário com a base normativa gravada.
        """
        vetor_formatado = "[" + ",".join(map(str, vetor_busca)) + "]"
        chunks_encontrados = []
        
        with self.obter_conexao() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_texto 
                    FROM embeddings_normativos 
                    ORDER BY vetor_embedding <=> %s 
                    LIMIT %s;
                    """,
                    (vetor_formatado, limite)
                )
                rows = cur.fetchall()
                for row in rows:
                    chunks_encontrados.append(row[0])
                    
        return chunks_encontrados
