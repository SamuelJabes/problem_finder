# Reddit Pain Point Analyzer 🔍

Descubra oportunidades de negócio através da análise automatizada de pain points no Reddit com análise avançada por clustering e LLM.

## 📋 Visão Geral

Este projeto coleta posts e comentários de subreddits relacionados a negócios e empreendedorismo, identifica automaticamente quais expressam dores/problemas dos usuários, utiliza clustering para agrupar problemas similares, e gera análises detalhadas de oportunidades de mercado usando LLM.

### 🎯 Subreddits Analisados
- Entrepreneur
- SaaS  
- NoStupidQuestions
- personalfinance
- smallbusiness
- socialmedia
- askatherapist
- productivity
- Accounting

Referência de escolhas: [aqui!](https://blog.venturemagazine.net/8-startup-ideas-that-keep-trending-on-reddit-with-analysis-f34b9a78455e)

## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone <seu-repo>
cd reddit-pain-analyzer
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente
```bash
cp .env.example .env
# Edite o arquivo .env com sua chave da OpenAI
```

**Variáveis necessárias no `.env`:**
```env
OPENAI_API_KEY=sua-chave-openai-aqui
```

## 💻 Como Usar

### 🎯 Pipeline Completo - Recomendado
Execute todo o processo de análise avançada de uma vez:
```bash
python reddit_analyzer.py full-pipeline
```

### 🔄 Pipelines Alternativos
```bash
# Pipeline básico (scrape + classify apenas)
python reddit_analyzer.py pipeline

# Continuar de dados existentes (pula scraping)
python reddit_analyzer.py continue
```

## 📚 Comandos Disponíveis

### 🔵 Comandos Básicos

#### 1. Scraping do Reddit
```bash
# Scraping básico (subreddits padrão)
python reddit_analyzer.py scrape

# Scraping customizado
python reddit_analyzer.py scrape \
  --subreddits "Entrepreneur,SaaS,productivity" \
  --max-pages 3 \
  --max-posts 10 \
  --include-comments
```

#### 2. Classificação de Pain Points
```bash
# Classificar textos coletados
python reddit_analyzer.py classify

# Especificar arquivos customizados
python reddit_analyzer.py classify \
  --input-file "data/comments/all_texts.json" \
  --output-file "data/comments/pain_points.json"
```

#### 3. Análise de Oportunidades Individuais
```bash
# Ver as melhores oportunidades de negócio
python reddit_analyzer.py insights

# Filtros customizados para oportunidades
python reddit_analyzer.py insights \
  --min-intensity 8 \
  --min-confidence 9 \
  --limit 5
```

### 🟢 Análise Avançada por Clustering

#### 4. Geração de Embeddings
```bash
# Gerar embeddings para clustering
python reddit_analyzer.py embeddings

# Especificar arquivos customizados
python reddit_analyzer.py embeddings \
  --input-file "data/comments/pain_points.json" \
  --output-file "data/embeddings/custom_embeddings.json"
```

#### 5. Clustering e Análise de Negócio
```bash
# Clustering com análise automática de oportunidades
python reddit_analyzer.py cluster

# Clustering customizado
python reddit_analyzer.py cluster \
  --n-clusters 5 \
  --embeddings-file "data/embeddings/reddit_embeddings.json"
```

#### 6. Visualização de Clusters
```bash
# Ver resumo de todos os clusters
python reddit_analyzer.py show-clusters

# Ver detalhes de um cluster específico
python reddit_analyzer.py show-clusters --cluster-id 0 --limit 15

# Ver detalhes de cluster com mais exemplos
python reddit_analyzer.py show-clusters --cluster-id 2 --limit 25
```

#### 7. Extração Detalhada de Insights
```bash
# Extrair insights detalhados em português (baseado no Jupyter original)
python reddit_analyzer.py extract

# Especificar arquivos customizados
python reddit_analyzer.py extract \
  --clusters-file "data/clusters/clusters.json" \
  --output-file "data/clusters/detailed_insights.txt"
```

### 🔵 Informações e Estatísticas
```bash
# Ver informações completas do projeto
python reddit_analyzer.py info

# Ver estatísticas dos dados coletados
python reddit_analyzer.py stats
```

## 📁 Estrutura dos Dados

```
data/
├── subreddits/
│   ├── Entrepreneur/
│   │   └── subreddit.json      # Posts do r/Entrepreneur
│   ├── SaaS/
│   │   └── subreddit.json      # Posts do r/SaaS
│   ├── NoStupidQuestions/
│   │   └── subreddit.json      # Posts do r/NoStupidQuestions
│   └── ...
├── comments/
│   ├── all_texts.json          # Todos os textos coletados
│   └── pain_points.json        # Pain points identificados com metadados
├── embeddings/
│   └── reddit_embeddings.json  # Embeddings 3072D para clustering
├── clusters/
│   ├── clusters.json           # Análise estruturada dos clusters
│   └── extracted_insights.txt  # Insights detalhados em português
```

## 📊 Exemplos de Saída

### 🎯 Pipeline Completo com Clustering
```bash
🚀 Starting COMPLETE pipeline (all steps)...

📡 Step 1: Scraping Reddit data...
✅ Scraping: 967 texts collected

🤖 Step 2: Classifying pain points...
✅ Classification: 342 pain points found

🧠 Step 3: Generating embeddings...
✅ Embeddings: 342 embeddings generated

🔍 Step 4: Clustering into 7 groups...
✅ Clustering completed! Found 7 clusters

🤖 Step 5: Analyzing clusters with LLM...
✅ Analysis completed!

📝 Step 6: Extracting detailed business insights...
✅ Insights extraction completed!

🎉 COMPLETE PIPELINE FINISHED!

🚀 TOP 3 BUSINESS OPPORTUNITIES:

🥇 #1 - Payroll Service Failures
   💡 Opportunity: Comprehensive payroll solution with guaranteed accuracy
   🎯 Target: Small to medium-sized businesses
   ⚡ Urgency: 9/10 | 📊 Items: 65

🥈 #2 - Ineffective Lead Management and Follow-Up
   💡 Opportunity: Lead management and CRM platform for SMBs
   🎯 Target: Service industries (plumbing, roofing, consulting)
   ⚡ Urgency: 8/10 | 📊 Items: 49

🥉 #3 - Social Media Business Challenges
   💡 Opportunity: Social media management platform focused on conversion
   🎯 Target: Small business owners, digital marketers
   ⚡ Urgency: 8/10 | 📊 Items: 62
```

### 🔍 Análise Detalhada de Clusters
```bash
python reddit_analyzer.py show-clusters --cluster-id 0

🔍 Cluster 0 Details
📊 Count: 65 pain points
🎯 Main Theme: Payroll Service Failures
🚀 Business Opportunity: Develop a comprehensive payroll solution that guarantees accuracy and proactive customer support with transparent communication
👥 Target Audience: Small to medium-sized businesses using payroll service providers
⚡ Urgency: 9/10
📈 Market Size: Large
🔧 Solution Complexity: Medium

📝 Common Problems:
   1. Payroll processing errors resulting in significant financial loss
   2. Lack of communication and customer service from payroll providers
   3. Frequent tax filing mistakes causing penalties
   4. Delayed problem resolution, leading to distrust in service providers
   5. High dependency on payroll software with little accountability

💬 Example Pain Points:
   1. "How a $47,000 payroll mistake almost killed my agency"
   2. "That's brutal. Payroll is the ONE thing you pay these services to handle..."
   3. "Talk to the IRS. They waived the penalty for the tax mistake my accountant..."
```

### 📋 Resumo de Todos os Clusters
```bash
python reddit_analyzer.py show-clusters

┏━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃ ID  ┃ Count ┃ Main Theme                   ┃ Business Opportunity                        ┃ Urgency ┃ Market  ┃
┡━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ 0   │ 65    │ Payroll Service Failures     │ Comprehensive payroll solution with...      │ 9/10    │ large   │
│ 2   │ 49    │ Ineffective Lead Management  │ Lead management and CRM platform...         │ 8/10    │ large   │
│ 4   │ 44    │ Entrepreneurial Challenges   │ Platform for small business owners...       │ 8/10    │ large   │
│ 5   │ 64    │ Business Operations Challenges│ Affordable SaaS platform with...          │ 8/10    │ large   │
│ 1   │ 26    │ Financial Literacy Issues    │ Online financial literacy training...       │ 8/10    │ large   │
│ 3   │ 62    │ Social Media Business Issues │ Social media management platform...         │ 8/10    │ large   │
│ 6   │ 40    │ Debt Management Problems     │ Financial advisory platform for debt...     │ 9/10    │ large   │
└─────┴───────┴──────────────────────────────┴─────────────────────────────────────────────┴─────────┴─────────┘
```

### 📝 Insights Detalhados Extraídos
```bash
python reddit_analyzer.py extract
cat data/clusters/extracted_insights.txt
```

```
REDDIT PAIN POINT ANALYZER - EXTRACTED INSIGHTS
================================================================================
Generated by extract command
Total clusters analyzed: 7
================================================================================

================================================================================
CLUSTER 0 - (65 pain points)
================================================================================
[Problemas com Serviços de Folha de Pagamento (Payroll Services)]

[Início da lista de tópicos]
1. Erros Graves em Processamento de Folha
   - Empresas relatam perdas financeiras significativas devido a erros de cálculo
   - Falhas no processamento automático causando pagamentos incorretos

2. Atendimento ao Cliente Deficiente
   - Dificuldade para contatar suporte quando problemas ocorrem
   - Demora excessiva na resolução de questões críticas

3. Problemas com Compliance Fiscal
   - Erros frequentes no recolhimento de impostos
   - Multas e penalidades por falhas do sistema de terceiros

4. Falta de Transparência e Comunicação
   - Usuários não são informados sobre mudanças ou problemas
   - Processos internos pouco claros para os clientes

5. Dependência Excessiva sem Accountability
   - Empresas ficam reféns de sistemas falhos
   - Pouca responsabilização por parte dos provedores
[Fim da lista de tópicos]

================================================================================
CLUSTER 2 - (49 pain points)
================================================================================
[Cluster: Ineficiências em Processos de Vendas e Atendimento ao Cliente]

[Início da lista de tópicos]
1. Falha no Follow-up de Leads
   - Tempo de resposta excessivo para novos prospects (23h vs 5min ideal)
   - Leads qualificados sendo perdidos por falta de acompanhamento

2. Foco Excessivo em Geração vs Conversão
   - Investimento alto em captar leads, baixo em nutrir existentes
   - ROI inadequado dos investimentos em marketing

3. Falta de Processos Estruturados
   - Ausência de sistemática clara para gestão de pipeline
   - Dependência de ações manuais e não automatizadas

4. Problemas de Motivação da Equipe
   - Time de vendas sem incentivos para melhorar follow-up
   - Cultura de "quantidade sobre qualidade" em leads

5. Subutilização de Dados Disponíveis
   - Empresas não aproveitam informações existentes sobre clientes
   - Falta de insights baseados em dados para otimização
[Fim da lista de tópicos]
```

### 💡 Oportunidades de Alto Impacto
```bash
python reddit_analyzer.py insights --min-intensity 8 --min-confidence 9

🚀 Top Business Opportunities Found: 23
   Criteria: Intensity ≥ 8, Confidence ≥ 9

📋 Opportunity #1
   Category: Business
   Intensity: 9/10
   Confidence: 10/10
   Pain Point: "How a $47,000 payroll mistake almost killed my agency"

📋 Opportunity #2
   Category: Financial
   Intensity: 9/10
   Confidence: 9/10
   Pain Point: "Should I sell my car to pay off $19k credit card debt..."

📊 Opportunity Categories:
   Business: 8
   Financial: 7
   Technical: 5
   Marketing: 3
```

## ⚙️ Configurações Avançadas

### Pipeline Customizado
```bash
python reddit_analyzer.py full-pipeline \
  --subreddits "Entrepreneur,SaaS,productivity,startups" \
  --max-pages 5 \
  --max-posts 15 \
  --n-clusters 10 \
  --api-key "sua-chave-openai"
```

### Clustering Avançado
```bash
# Gerar embeddings customizados
python reddit_analyzer.py embeddings \
  --input-file "data/comments/filtered_pain_points.json"

# Clustering com mais grupos
python reddit_analyzer.py cluster \
  --n-clusters 12 \
  --embeddings-file "data/embeddings/reddit_embeddings.json"

# Análise focada em clusters específicos
python reddit_analyzer.py show-clusters --cluster-id 3 --limit 50
```

## 🛠️ Arquitetura Técnica

### Classes Principais
- `RedditScraper`: [Web scraping assíncrono do Reddit](https://scrapfly.io/blog/how-to-scrape-reddit-social-data/)
- `PainPointClassifier`: Classificação de pain points com GPT-4o
- `EmbeddingGenerator`: Geração de embeddings com OpenAI
- `ClusterAnalyzer`: Clustering K-means + análise com LLM
- Interface CLI robusta com Typer + Rich

### Modelos Utilizados
- **Classificação**: GPT-4o (gpt-4o-2024-08-06)
- **Embeddings**: text-embedding-3-large (3072 dimensões)
- **Clustering**: K-means scikit-learn
- **Análise**: GPT-4o com structured output

### Schemas Pydantic
- `ClassifySchema`: Estrutura para classificação de pain points
- `ClusterAnalysisSchema`: Estrutura para análise de clusters
- `RedditPost/Comment`: Estruturas para dados do Reddit
- `PainPoint`: Estrutura para pain points identificados

## 🔄 Workflows Recomendados

### 1. Análise Completa (Recomendado)
```bash
# Uma única execução para análise completa
python reddit_analyzer.py full-pipeline

# Visualizar resultados
python reddit_analyzer.py show-clusters
python reddit_analyzer.py insights --min-intensity 7
cat data/clusters/extracted_insights.txt
```

### 2. Análise Iterativa
```bash
# Fase 1: Coleta de dados
python reddit_analyzer.py scrape --max-pages 3

# Fase 2: Identificação de problemas  
python reddit_analyzer.py classify

# Fase 3: Análise avançada
python reddit_analyzer.py embeddings
python reddit_analyzer.py cluster --n-clusters 8

# Fase 4: Extração de insights
python reddit_analyzer.py extract
python reddit_analyzer.py show-clusters
```

### 3. Análise Focada
```bash
# Focar em subreddits específicos
python reddit_analyzer.py scrape --subreddits "SaaS,Entrepreneur"
python reddit_analyzer.py continue

# Análise com filtros rigorosos
python reddit_analyzer.py insights --min-intensity 8 --min-confidence 9
```

## Vídeo de Apresentação
> Assista aqui ao vídeo.

[![Vídeo de Apresentação](https://img.youtube.com/vi/2HiQ7tl4-rA/0.jpg)](https://youtu.be/2HiQ7tl4-rA)

## Estudo da solução

Na pasta

```
src/
├── impacto_solucao.md
├── outras_alternativas.md
```

Estão alguns relatos e estudos sobre a aplicação. Em `impacto_solucao.md` nós trazemos uma sumarização e noção de impacto da nossa aplicação por meio de entrevistas
reais com empreendedores do HUB de Inovação do Insper. Já em `outras_alternativas` trazemos um estudo de outras soluções que já existiam e como a nossa pode se diferenciar. 