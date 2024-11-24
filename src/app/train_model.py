from vanna.ollama import Ollama
from vanna.qdrant import Qdrant_VectorStore
from qdrant_client import QdrantClient
from vanna.flask import VannaFlaskApp

class MyVanna(Qdrant_VectorStore, Ollama):
    def __init__(self, config=None):
        
        if config is None:
            config = {}
            
            
        qdrant_client = QdrantClient(url="localhost", port=6333)
        
        qdrant_config = {'client': qdrant_client}
        ollama_config = {
            'model': 'llama2:7b',  # Specify the model name
            'url': 'http://localhost:11434',  # Default Ollama API endpoint
            'temperature': 0.7,  # Optional: adjust temperature for response randomness
            'num_ctx': 4096,     # Context window size
            'num_thread': 4      # Number of threads to use
        }
        
        config = {**qdrant_config, **ollama_config, **config}
        Qdrant_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


if __name__ == "__main__":
    vn = MyVanna()

    vn.connect_to_postgres(host='localhost', dbname='geekmaster', user='admin', password='admin', port='5432')

    # The information schema query may need some tweaking depending on your database. This is a good starting point.
    df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = 'public' ")

    # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
    plan = vn.get_training_plan_generic(df_information_schema)

    # If you like the plan, then uncomment this and run it to train
    vn.train(plan=plan)

    vn.train(
    question="Quais são os 5 vendedores que mais venderam em valor total?",
    sql="""
    SELECT v.nome, COUNT(c.id_compra) as total_vendas, SUM(c.valor_total) as valor_total
    FROM vendedor v
    JOIN compras c ON v.id_vendedor = c.id_vendedor
    GROUP BY v.nome
    ORDER BY valor_total DESC
    LIMIT 5;
    """
    )

    vn.train(
    question="Qual é o ticket médio de compras por cliente?",
    sql="""
    SELECT cl.nome, ROUND(AVG(c.valor_total), 2) as ticket_medio
    FROM clientes cl
    JOIN compras c ON cl.id_cliente = c.id_cliente
    GROUP BY cl.nome
    ORDER BY ticket_medio DESC;
    """
    )

    vn.train(
    question="Quais produtos têm estoque menor que 10 unidades?",
    sql="""
    SELECT nome, estoque, preco_unitario
    FROM produtos
    WHERE estoque < 10
    ORDER BY estoque ASC;
    """
    )

    vn.train(
    question="Quantas vendas foram realizadas nos últimos 30 dias?",
    sql="""
    SELECT COUNT(*) as total_vendas, SUM(valor_total) as valor_total
    FROM compras
    WHERE data_compra >= CURRENT_DATE - INTERVAL '30 days';
    """
    )

    vn.train(
    question="Quais são os clientes que não fizeram nenhuma compra?",
    sql="""
    SELECT cl.nome, cl.email, cl.data_cadastro
    FROM clientes cl
    LEFT JOIN compras c ON cl.id_cliente = c.id_cliente
    WHERE c.id_compra IS NULL;
    """
    )

    vn.train(
    question="Qual é a média de produtos por compra?",
    sql="""
    SELECT ROUND(AVG(quantidade), 2) as media_produtos_por_compra
    FROM compras;
    """
    )

    vn.train(
    question="Quais são os produtos mais rentáveis?",
    sql="""
    SELECT p.nome, SUM(c.quantidade) as quantidade_vendida, 
            SUM(c.valor_total) as receita_total
    FROM produtos p
    JOIN compras c ON p.id_produto = c.id_produto
    GROUP BY p.nome
    ORDER BY receita_total DESC
    LIMIT 10;
    """
    )

    vn.train(
    question="Qual é o histórico de vendas por mês?",
    sql="""
    SELECT DATE_TRUNC('month', data_compra) as mes,
            COUNT(*) as total_vendas,
            SUM(valor_total) as valor_total
    FROM compras
    GROUP BY DATE_TRUNC('month', data_compra)
    ORDER BY mes DESC;
    """
    )

    vn.train(
    question="Quais vendedores têm mais clientes associados?",
    sql="""
    SELECT v.nome, COUNT(DISTINCT cl.id_cliente) as total_clientes
    FROM vendedor v
    LEFT JOIN clientes cl ON v.id_vendedor = cl.id_vendedor
    GROUP BY v.nome
    ORDER BY total_clientes DESC;
    """
    )

    vn.train(
    question="Qual é a distribuição de vendas por dia da semana?",
    sql="""
    SELECT EXTRACT(DOW FROM data_compra) as dia_semana,
            COUNT(*) as total_vendas,
            SUM(valor_total) as valor_total
    FROM compras
    GROUP BY dia_semana
    ORDER BY dia_semana;
    """
    )

    vn.train(
    question="Quais são os clientes que gastaram mais de 5000 reais?",
    sql="""
    SELECT cl.nome, SUM(c.valor_total) as total_gasto
    FROM clientes cl
    JOIN compras c ON cl.id_cliente = c.id_cliente
    GROUP BY cl.nome
    HAVING SUM(c.valor_total) > 5000
    ORDER BY total_gasto DESC;
    """
    )

    vn.train(
    question="Qual é o tempo médio entre compras por cliente?",
    sql="""
    WITH ComprasDiff AS (
        SELECT id_cliente,
                data_compra,
                LAG(data_compra) OVER (PARTITION BY id_cliente ORDER BY data_compra) as compra_anterior
        FROM compras
    )
    SELECT cl.nome,
            AVG(data_compra - compra_anterior) as tempo_medio_entre_compras
    FROM ComprasDiff cd
    JOIN clientes cl ON cd.id_cliente = cl.id_cliente
    WHERE compra_anterior IS NOT NULL
    GROUP BY cl.nome;
    """
    )

    vn.train(
    question="Quais produtos nunca foram vendidos?",
    sql="""
    SELECT p.nome, p.preco_unitario, p.estoque
    FROM produtos p
    LEFT JOIN compras c ON p.id_produto = c.id_produto
    WHERE c.id_compra IS NULL;
    """
    )

    vn.train(
    question="Qual é o ranking de vendedores por quantidade de vendas no último mês?",
    sql="""
    SELECT v.nome,
            COUNT(c.id_compra) as total_vendas,
            SUM(c.valor_total) as valor_total
    FROM vendedor v
    LEFT JOIN compras c ON v.id_vendedor = c.id_vendedor
    WHERE c.data_compra >= DATE_TRUNC('month', CURRENT_DATE)
    GROUP BY v.nome
    ORDER BY total_vendas DESC;
    """
    )

    vn.train(
    question="Quais são os horários de pico de vendas?",
    sql="""
    SELECT EXTRACT(HOUR FROM data_compra) as hora,
            COUNT(*) as total_vendas,
            SUM(valor_total) as valor_total
    FROM compras
    GROUP BY hora
    ORDER BY total_vendas DESC;
    """
    )

    vn.train(
    question="Qual é a taxa de conversão de cada vendedor (vendas/total de clientes)?",
    sql="""
    SELECT v.nome,
            COUNT(DISTINCT c.id_compra) as total_vendas,
            COUNT(DISTINCT cl.id_cliente) as total_clientes,
            ROUND(COUNT(DISTINCT c.id_compra)::decimal / 
                NULLIF(COUNT(DISTINCT cl.id_cliente), 0), 2) as taxa_conversao
    FROM vendedor v
    LEFT JOIN clientes cl ON v.id_vendedor = cl.id_vendedor
    LEFT JOIN compras c ON v.id_vendedor = c.id_vendedor
    GROUP BY v.nome
    ORDER BY taxa_conversao DESC;
    """
    )

    vn.train(
    question="Quais clientes fizeram compras em todos os meses dos últimos 6 meses?",
    sql="""
    WITH MesesCliente AS (
        SELECT id_cliente,
                COUNT(DISTINCT DATE_TRUNC('month', data_compra)) as meses_distintos
        FROM compras
        WHERE data_compra >= CURRENT_DATE - INTERVAL '6 months'
        GROUP BY id_cliente
        HAVING COUNT(DISTINCT DATE_TRUNC('month', data_compra)) = 6
    )
    SELECT cl.nome, cl.email
    FROM clientes cl
    JOIN MesesCliente mc ON cl.id_cliente = mc.id_cliente;
    """
    )

    vn.train(
    question="Qual é o valor médio de compra por faixa horária?",
    sql="""
    SELECT EXTRACT(HOUR FROM data_compra) as hora,
            ROUND(AVG(valor_total), 2) as valor_medio,
            COUNT(*) as total_vendas
    FROM compras
    GROUP BY hora
    ORDER BY hora;
    """
    )

    vn.train(
    question="Quais são os produtos com maior margem de vendas?",
    sql="""
    SELECT p.nome,
            p.preco_unitario,
            COUNT(c.id_compra) as total_vendas,
            SUM(c.valor_total) as receita_total,
            ROUND(AVG(c.valor_total/c.quantidade), 2) as preco_medio_venda
    FROM produtos p
    JOIN compras c ON p.id_produto = c.id_produto
    GROUP BY p.nome, p.preco_unitario
    ORDER BY total_vendas DESC;
    """
    )

    vn.train(
    question="Qual é a sazonalidade das vendas por mês do ano?",
    sql="""
    SELECT EXTRACT(MONTH FROM data_compra) as mes,
            EXTRACT(YEAR FROM data_compra) as ano,
            COUNT(*) as total_vendas,
            SUM(valor_total) as valor_total
    FROM compras
    GROUP BY mes, ano
    ORDER BY ano, mes;
    """
    )

    vn.train(
    question="Quais clientes fizeram compras acima da média geral?",
    sql="""
    WITH MediaGeral AS (
        SELECT AVG(valor_total) as media
        FROM compras
    )
    SELECT cl.nome, SUM(c.valor_total) as total_gasto, mg.media as media_geral
    FROM clientes cl
    JOIN compras c ON cl.id_cliente = c.id_cliente
    CROSS JOIN MediaGeral mg
    GROUP BY cl.nome, mg.media
    HAVING SUM(c.valor_total) > mg.media
    ORDER BY total_gasto DESC;
    """
    )

    vn.train(
    question="Qual é a frequência de compra dos clientes nos últimos 3 meses?",
    sql="""
    SELECT cl.nome,
            COUNT(c.id_compra) as total_compras,
            COUNT(c.id_compra)::float / 3 as media_mensal
    FROM clientes cl
    LEFT JOIN compras c ON cl.id_cliente = c.id_cliente
    WHERE c.data_compra >= CURRENT_DATE - INTERVAL '3 months'
    GROUP BY cl.nome
    ORDER BY media_mensal DESC;
    """
    )

    vn.train(
    question="Quais produtos tiveram aumento nas vendas mês a mês?",
    sql="""
    WITH VendasMensais AS (
        SELECT 
            p.id_produto,
            p.nome,
            DATE_TRUNC('month', c.data_compra) as mes,
            SUM(c.quantidade) as qtd_vendida,
            LAG(SUM(c.quantidade)) OVER (PARTITION BY p.id_produto ORDER BY DATE_TRUNC('month', c.data_compra)) as qtd_anterior
        FROM produtos p
        JOIN compras c ON p.id_produto = c.id_produto
        GROUP BY p.id_produto, p.nome, mes
    )
    SELECT nome, mes, qtd_vendida, qtd_anterior,
            ROUND(((qtd_vendida - qtd_anterior) / qtd_anterior::float * 100), 2) as crescimento_percentual
    FROM VendasMensais
    WHERE qtd_anterior IS NOT NULL
    AND qtd_vendida > qtd_anterior
    ORDER BY mes DESC, crescimento_percentual DESC;
    """
    )

    vn.train(
    question="Qual é o perfil de compra dos clientes por valor?",
    sql="""
    SELECT 
        CASE 
            WHEN valor_total <= 100 THEN 'Até R\$100'
            WHEN valor_total <= 500 THEN 'R\$101 a R\$500'
            WHEN valor_total <= 1000 THEN 'R\$501 a R\$1000'
            ELSE 'Acima de R\$1000'
        END as faixa_valor,
        COUNT(*) as quantidade_compras,
        COUNT(DISTINCT id_cliente) as quantidade_clientes
    FROM compras
    GROUP BY faixa_valor
    ORDER BY MIN(valor_total);
    """
    )

    vn.train(
    question="Quais vendedores têm a maior taxa de recompra de clientes?",
    sql="""
    WITH ClientesRecorrentes AS (
        SELECT 
            v.id_vendedor,
            COUNT(DISTINCT c.id_cliente) as total_clientes,
            COUNT(DISTINCT CASE WHEN compras_cliente > 1 THEN c.id_cliente END) as clientes_recorrentes
        FROM vendedor v
        JOIN compras c ON v.id_vendedor = c.id_vendedor
        JOIN (
            SELECT id_cliente, COUNT(*) as compras_cliente
            FROM compras
            GROUP BY id_cliente
        ) cc ON c.id_cliente = cc.id_cliente
        GROUP BY v.id_vendedor
    )
    SELECT 
        v.nome,
        cr.total_clientes,
        cr.clientes_recorrentes,
        ROUND((cr.clientes_recorrentes::float / NULLIF(cr.total_clientes, 0) * 100), 2) as taxa_recompra
    FROM ClientesRecorrentes cr
    JOIN vendedor v ON cr.id_vendedor = v.id_vendedor
    ORDER BY taxa_recompra DESC;
    """
    )

    vn.train(
    question="Qual é o tempo médio entre a primeira e última compra dos clientes?",
    sql="""
    SELECT 
        cl.nome,
        MIN(c.data_compra) as primeira_compra,
        MAX(c.data_compra) as ultima_compra,
        MAX(c.data_compra) - MIN(c.data_compra) as tempo_entre_compras
    FROM clientes cl
    JOIN compras c ON cl.id_cliente = c.id_cliente
    GROUP BY cl.nome
    HAVING COUNT(c.id_compra) > 1
    ORDER BY tempo_entre_compras DESC;
    """
    )

    vn.train(
    question="Quais produtos são frequentemente comprados juntos?",
    sql="""
    WITH ComprasSimultaneas AS (
        SELECT 
            c1.id_produto as produto1,
            c2.id_produto as produto2,
            COUNT(*) as frequencia
        FROM compras c1
        JOIN compras c2 ON c1.id_cliente = c2.id_cliente 
            AND c1.data_compra = c2.data_compra 
            AND c1.id_produto < c2.id_produto
        GROUP BY c1.id_produto, c2.id_produto
    )
    SELECT 
        p1.nome as produto1,
        p2.nome as produto2,
        cs.frequencia
    FROM ComprasSimultaneas cs
    JOIN produtos p1 ON cs.produto1 = p1.id_produto
    JOIN produtos p2 ON cs.produto2 = p2.id_produto
    ORDER BY cs.frequencia DESC
    LIMIT 10;
    """
    )

    vn.train(
    question="Qual é a distribuição de vendas por dia do mês?",
    sql="""
    SELECT 
        EXTRACT(DAY FROM data_compra) as dia_mes,
        COUNT(*) as total_vendas,
        SUM(valor_total) as valor_total,
        ROUND(AVG(valor_total), 2) as ticket_medio
    FROM compras
    GROUP BY dia_mes
    ORDER BY dia_mes;
    """
    )

    vn.train(
    question="Quais vendedores têm o maior ticket médio?",
    sql="""
    SELECT 
        v.nome,
        COUNT(c.id_compra) as total_vendas,
        ROUND(AVG(c.valor_total), 2) as ticket_medio,
        MIN(c.valor_total) as menor_venda,
        MAX(c.valor_total) as maior_venda
    FROM vendedor v
    JOIN compras c ON v.id_vendedor = c.id_vendedor
    GROUP BY v.nome
    ORDER BY ticket_medio DESC;
    """
    )

    vn.train(
    question="Qual é a análise de cohort dos clientes por mês de cadastro?",
    sql="""
    WITH PrimeiraCompra AS (
        SELECT 
            id_cliente,
            DATE_TRUNC('month', MIN(data_compra)) as cohort_month
        FROM compras
        GROUP BY id_cliente
    ),
    ComprasPorMes AS (
        SELECT 
            pc.id_cliente,
            pc.cohort_month,
            DATE_TRUNC('month', c.data_compra) as purchase_month,
            COUNT(DISTINCT c.id_compra) as num_compras
        FROM PrimeiraCompra pc
        JOIN compras c ON pc.id_cliente = c.id_cliente
        GROUP BY pc.id_cliente, pc.cohort_month, DATE_TRUNC('month', c.data_compra)
    )
    SELECT 
        cohort_month,
        COUNT(DISTINCT id_cliente) as total_clientes,
        COUNT(DISTINCT CASE WHEN purchase_month > cohort_month THEN id_cliente END) as clientes_retornaram
    FROM ComprasPorMes
    GROUP BY cohort_month
    ORDER BY cohort_month;
    """
    )

    vn.train(
    question="Quais são os produtos com maior variação de preço nas vendas?",
    sql="""
    SELECT 
        p.nome,
        MIN(c.valor_total/c.quantidade) as menor_preco,
        MAX(c.valor_total/c.quantidade) as maior_preco,
        ROUND(AVG(c.valor_total/c.quantidade), 2) as preco_medio,
        ROUND(STDDEV(c.valor_total/c.quantidade), 2) as desvio_padrao
    FROM produtos p
    JOIN compras c ON p.id_produto = c.id_produto
    GROUP BY p.nome
    HAVING COUNT(c.id_compra) > 5
    ORDER BY desvio_padrao DESC;
    """
    )

    vn.train(
    question="Qual é a análise de vendas por estação do ano?",
    sql="""
    SELECT 
        CASE 
            WHEN EXTRACT(MONTH FROM data_compra) IN (12,1,2) THEN 'Verão'
            WHEN EXTRACT(MONTH FROM data_compra) IN (3,4,5) THEN 'Outono'
            WHEN EXTRACT(MONTH FROM data_compra) IN (6,7,8) THEN 'Inverno'
            ELSE 'Primavera'
        END as estacao,
        COUNT(*) as total_vendas,
        ROUND(AVG(valor_total), 2) as ticket_medio,
        SUM(valor_total) as valor_total
    FROM compras
    GROUP BY estacao
    ORDER BY MIN(EXTRACT(MONTH FROM data_compra));
    """
    )

    vn.train(
    question="Quais clientes têm o maior percentual de compras canceladas?",
    sql="""
    SELECT 
        cl.nome,
        COUNT(*) as total_compras,
        SUM(CASE WHEN c.status = 'cancelado' THEN 1 ELSE 0 END) as compras_canceladas,
        ROUND(SUM(CASE WHEN c.status = 'cancelado' THEN 1 ELSE 0 END)::float / COUNT(*) * 100, 2) as percentual_cancelamento
    FROM clientes cl
    JOIN compras c ON cl.id_cliente = c.id_cliente
    GROUP BY cl.nome
    HAVING COUNT(*) >= 5
    ORDER BY percentual_cancelamento DESC;
    """
    )

    vn.train(
    question="Qual é a análise de vendas por categoria de produto?",
    sql="""
    SELECT 
        p.categoria,
        COUNT(c.id_compra) as total_vendas,
        SUM(c.quantidade) as quantidade_vendida,
        ROUND(AVG(c.valor_total), 2) as ticket_medio,
        SUM(c.valor_total) as valor_total
    FROM produtos p
    JOIN compras c ON p.id_produto = c.id_produto
    GROUP BY p.categoria
    ORDER BY valor_total DESC;
    """
    )

    vn.train(
    question="Quais são os clientes que mais compraram em cada categoria de produto?",
    sql="""
    WITH RankingClientes AS (
        SELECT 
            cl.nome,
            p.categoria,
            COUNT(*) as total_compras,
            SUM(c.valor_total) as valor_total,
            RANK() OVER (PARTITION BY p.categoria ORDER BY COUNT(*) DESC) as ranking
        FROM clientes cl
        JOIN compras c ON cl.id_cliente = c.id_cliente
        JOIN produtos p ON c.id_produto = p.id_produto
        GROUP BY cl.nome, p.categoria
    )
    SELECT 
        categoria,
        nome as cliente,
        total_compras,
        valor_total
    FROM RankingClientes
    WHERE ranking = 1
    ORDER BY valor_total DESC;
    """
    )

    vn.train(
    question="Qual é a análise de vendas por faixa etária dos clientes?",
    sql="""
    SELECT 
        CASE 
            WHEN cl.idade < 25 THEN 'Até 25 anos'
            WHEN cl.idade BETWEEN 25 AND 35 THEN '25-35 anos'
            WHEN cl.idade BETWEEN 36 AND 50 THEN '36-50 anos'
            ELSE 'Acima de 50 anos'
        END as faixa_etaria,
        COUNT(DISTINCT cl.id_cliente) as total_clientes,
        COUNT(c.id_compra) as total_compras,
        ROUND(AVG(c.valor_total), 2) as ticket_medio,
        SUM(c.valor_total) as valor_total
    FROM clientes cl
    JOIN compras c ON cl.id_cliente = c.id_cliente
    GROUP BY faixa_etaria
    ORDER BY MIN(cl.idade);
    """
    )

    vn.train(
    question="Quais são os produtos mais vendidos por região?",
    sql="""
    SELECT 
        cl.regiao,
        p.nome as produto,
        COUNT(*) as total_vendas,
        SUM(c.quantidade) as quantidade_vendida,
        SUM(c.valor_total) as valor_total,
        RANK() OVER (PARTITION BY cl.regiao ORDER BY COUNT(*) DESC) as ranking
    FROM clientes cl
    JOIN compras c ON cl.id_cliente = c.id_cliente
    JOIN produtos p ON c.id_produto = p.id_produto
    GROUP BY cl.regiao, p.nome
    HAVING RANK() OVER (PARTITION BY cl.regiao ORDER BY COUNT(*) DESC) <= 3
    ORDER BY cl.regiao, ranking;
    """
    )

    vn.train(
    question="Qual é a análise de vendas por método de pagamento?",
    sql="""
    SELECT 
        c.metodo_pagamento,
        COUNT(*) as total_transacoes,
        ROUND(AVG(valor_total), 2) as valor_medio,
        SUM(valor_total) as valor_total,
        COUNT(DISTINCT id_cliente) as clientes_unicos
    FROM compras c
    GROUP BY c.metodo_pagamento
    ORDER BY valor_total DESC;
    """
    )

    vn.train(
    question="Quais são os vendedores com maior taxa de crescimento nas vendas?",
    sql="""
    WITH VendasMensais AS (
        SELECT 
            v.id_vendedor,
            v.nome,
            DATE_TRUNC('month', c.data_compra) as mes,
            SUM(c.valor_total) as valor_total,
            LAG(SUM(c.valor_total)) OVER (PARTITION BY v.id_vendedor ORDER BY DATE_TRUNC('month', c.data_compra)) as valor_anterior
        FROM vendedor v
        JOIN compras c ON v.id_vendedor = c.id_vendedor
        GROUP BY v.id_vendedor, v.nome, mes
    )
    SELECT 
        nome,
        mes,
        valor_total,
        valor_anterior,
        ROUND(((valor_total - valor_anterior) / valor_anterior * 100), 2) as crescimento_percentual
    FROM VendasMensais
    WHERE valor_anterior IS NOT NULL
    ORDER BY crescimento_percentual DESC;
    """
    )
    
    vn.train(
    ddl="""
    CREATE TABLE vendedor (
        id_vendedor SERIAL PRIMARY KEY,
        nome VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        data_contratacao DATE NOT NULL,
        status BOOLEAN DEFAULT true
    );
    """
    )

    vn.train(
    ddl="""
    CREATE TABLE clientes (
        id_cliente SERIAL PRIMARY KEY,
        nome VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        telefone VARCHAR(20),
        data_cadastro DATE NOT NULL,
        id_vendedor INTEGER REFERENCES vendedor(id_vendedor)
    );
    """
    )

    vn.train(
    ddl="""
    CREATE TABLE produtos (
        id_produto SERIAL PRIMARY KEY,
        nome VARCHAR(100) NOT NULL,
        descricao TEXT,
        preco_unitario DECIMAL(10,2) NOT NULL,
        estoque INTEGER NOT NULL,
        data_cadastro DATE NOT NULL
    );
    """
    )

    vn.train(
    ddl="""
    CREATE TABLE compras (
        id_compra SERIAL PRIMARY KEY,
        id_cliente INTEGER REFERENCES clientes(id_cliente),
        id_produto INTEGER REFERENCES produtos(id_produto),
        id_vendedor INTEGER REFERENCES vendedor(id_vendedor),
        quantidade INTEGER NOT NULL,
        valor_total DECIMAL(10,2) NOT NULL,
        data_compra TIMESTAMP NOT NULL
    );
    """
    )

    # Treinamento com os índices
    vn.train(
    ddl="""
    CREATE INDEX idx_compras_cliente ON compras(id_cliente);
    CREATE INDEX idx_compras_produto ON compras(id_produto);
    CREATE INDEX idx_compras_vendedor ON compras(id_vendedor);
    CREATE INDEX idx_clientes_vendedor ON clientes(id_vendedor);
    """
    )

    # Treinamento com as constraints de chave estrangeira
    vn.train(
    ddl="""
    ALTER TABLE clientes
    ADD CONSTRAINT fk_cliente_vendedor
    FOREIGN KEY (id_vendedor)
    REFERENCES vendedor(id_vendedor);

    ALTER TABLE compras
    ADD CONSTRAINT fk_compra_cliente
    FOREIGN KEY (id_cliente)
    REFERENCES clientes(id_cliente);

    ALTER TABLE compras
    ADD CONSTRAINT fk_compra_produto
    FOREIGN KEY (id_produto)
    REFERENCES produtos(id_produto);

    ALTER TABLE compras
    ADD CONSTRAINT fk_compra_vendedor
    FOREIGN KEY (id_vendedor)
    REFERENCES vendedor(id_vendedor);
    """
    )

    # Treinamento com as constraints de validação
    vn.train(
    ddl="""
    ALTER TABLE produtos
    ADD CONSTRAINT check_preco_positivo
    CHECK (preco_unitario > 0);

    ALTER TABLE produtos
    ADD CONSTRAINT check_estoque_positivo
    CHECK (estoque >= 0);

    ALTER TABLE compras
    ADD CONSTRAINT check_quantidade_positiva
    CHECK (quantidade > 0);

    ALTER TABLE compras
    ADD CONSTRAINT check_valor_total_positivo
    CHECK (valor_total > 0);
    """
    )

    # Treinamento com os comentários das tabelas
    vn.train(
    ddl="""
    COMMENT ON TABLE vendedor IS 'Tabela que armazena informações dos vendedores da empresa';
    COMMENT ON TABLE clientes IS 'Tabela que armazena informações dos clientes';
    COMMENT ON TABLE produtos IS 'Tabela que armazena o catálogo de produtos disponíveis';
    COMMENT ON TABLE compras IS 'Tabela que registra todas as transações de compras realizadas';
    """
    )

    # Treinamento com os comentários das colunas principais
    vn.train(
    ddl="""
    COMMENT ON COLUMN vendedor.id_vendedor IS 'Identificador único do vendedor';
    COMMENT ON COLUMN clientes.id_cliente IS 'Identificador único do cliente';
    COMMENT ON COLUMN produtos.id_produto IS 'Identificador único do produto';
    COMMENT ON COLUMN compras.id_compra IS 'Identificador único da compra';
    """
    )
    
    print("Treinamento finalizado com sucesso!")