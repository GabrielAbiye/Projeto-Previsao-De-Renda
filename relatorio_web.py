import streamlit as st
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt

sns.set(context='talk', style='whitegrid')
st.set_page_config(page_title= 'Analise de renda' , layout='centered')

def categorizacao (renda):
    categoria = []
    for i in renda:
        if i <= 2800:
            categoria.append('Classe E')
        elif 2800 >= i <= 5600:
            categoria.append('Classe D')
        elif 5600 >= i <= 14100:
            categoria.append('Classe C')
        elif 14100 >= i <= 28800:
            categoria.append('Classe B')
        elif i > 28800:
            categoria.append('Classe A')
        else:
            categoria.append('Renda não informada')

    return categoria

renda = pd.read_csv(r"C:\Users\g_abi\OneDrive\Documentos\Data Science\Data Science EBAC\Módulo 16\Projeto\projeto 2\input\previsao_de_renda.csv")
renda = (renda.assign(data_ref = lambda x : pd.to_datetime(x.data_ref))
        .pipe(lambda x : x.drop_duplicates(subset= 'id_cliente' , keep = 'first'))
        .assign(tempo_emprego = lambda x : x.tempo_emprego.fillna(0))
        .pipe(lambda x : x.drop(columns= 'Unnamed: 0'))
)
renda['classe_renda'] = categorizacao(renda.renda)
# Definindo as páginas
pages = {
    'PÁGINA INICIAL' : 1,
    'BASE DE DADOS': 2,
    'ANÁLISE DAS VÁRIAVEIS CATEGÓRICAS' : 3,
    'CORRELAÇÃO ENTRE VARIÁVEIS QUANTITATIVAS' : 4,
    'ANÁLISES TEMPORAIS' : 5
}

# Menu lateral
selected_page = st.sidebar.radio("Select a page:", list(pages.keys()))

# Mostrando a página selecionada
if selected_page == 'PÁGINA INICIAL':
    st.title('PÁGINA INICIAL')
    st.subheader('Sobre mim' , divider= 'blue')
    st.write('Olá, seja bem vindo. Me chamo Gabriel Abiyê , sou formado em Administração Empresarial pela Universidade Salvador e pós-graduado em Business Intelligence pela Faculdade Estácio de Sá. Atualmente, estou me especializando em Ciência de Dados pela EBAC. Com um forte background em administração e inteligência de negócios, estou agora focado em aprofundar meus conhecimentos e habilidades na área de ciência de dados, buscando aplicar técnicas avançadas de análise e machine learning para resolver problemas complexos e tomar decisões informadas.')
    st.write('Este é um projeto para previsão de renda utilizando um conjunto de dados com diversas variáveis demográficas e financeiras dos clientes. Este projeto tem como objetivo principal a construção de um modelo preditivo que identifique padrões e relações entre essas variáveis e a renda dos clientes, proporcionando uma análise aprofundada e precisa. Embora ainda esteja em estágio inicial, o foco é aplicar e consolidar conhecimentos teóricos na prática, aprimorando habilidades na área de machine learning e análise de dados. Esta página do Streamlit é um complemento do meu notebook Python, que está disponível no meu GitHub:')
    st.markdown('[GabrielAbiye](https://github.com/GabrielAbiye)')
    st.subheader('Sobre a base de dados' , divider = 'blue')
    st.write('O conjunto de dados utilizado neste projeto contém informações demográficas e financeiras detalhadas de clientes, que são essenciais para a construção de um modelo preditivo de renda. Abaixo estão as descrições de cada variável')
    st.markdown("""
O conjunto de dados utilizado neste projeto contém informações demográficas e financeiras dos clientes, essenciais para construir um modelo preditivo de renda. Abaixo, apresento a descrição das variáveis presentes no DataFrame:

1. **data_ref**: Esta variável indica a data de referência na qual as informações foram coletadas. Serve para rastrear a temporalidade dos dados.

2. **id_cliente**: Um identificador único para cada cliente no conjunto de dados, facilitando o rastreamento e a análise individual dos dados.

3. **sexo**: Representa o sexo do cliente, categorizado como masculino ou feminino. É uma variável que ajuda a entender como a renda pode variar entre diferentes gêneros.

4. **posse_de_veiculo**: Indica se o cliente possui um veículo (True) ou não (False). Esta variável booleana é útil para analisar a relação entre a posse de um veículo e a renda.

5. **posse_de_imovel**: Indica se o cliente possui um imóvel (True) ou não (False). Semelhante à posse de veículo, esta variável booleana ajuda a entender a correlação entre a posse de imóvel e a renda.

6. **qtd_filhos**: Refere-se à quantidade de filhos que o cliente possui. Essa variável numérica pode influenciar as despesas e, portanto, a renda disponível do cliente.

7. **tipo_renda**: Categorização da fonte de renda do cliente, como assalariado, autônomo, aposentado, entre outros. Esta variável é importante para identificar diferentes padrões de renda.

8. **educacao**: Grau de instrução do cliente, variando de ensino fundamental a superior. O nível educacional pode ter uma influência significativa sobre a renda.

9. **estado_civil**: Estado civil do cliente, como solteiro, casado ou divorciado. O estado civil pode impactar a renda devido a diferentes responsabilidades financeiras.

10. **tipo_residencia**: Tipo de residência do cliente, que pode ser própria, alugada, financiada, etc. Esta variável ajuda a analisar como o tipo de residência está associado à renda.

11. **idade**: Idade do cliente em anos. A idade pode estar relacionada à fase da carreira e, consequentemente, à renda.

12. **tempo_emprego**: Tempo que o cliente está no emprego atual, expresso em anos. A experiência no emprego pode afetar a renda do cliente.

13. **qt_pessoas_residencia**: Número de pessoas que residem com o cliente. Esta variável pode refletir a composição familiar e seu impacto na renda.

14. **renda**: A renda mensal do cliente em reais, que é a variável alvo deste projeto. O objetivo é prever essa variável com base nas demais informações.

Cada uma dessas variáveis fornece informações cruciais para a análise e a construção de um modelo preditivo eficaz.
""")
    
elif selected_page == "BASE DE DADOS":
    st.title('BASE DE DADOS')
    st.subheader("Tratamento de Dados Faltantes")

    st.markdown("""
    A base de dados utilizada neste projeto apresentava uma quantidade significativa de dados faltantes. Para garantir que as análises fossem precisas e confiáveis, foi necessário realizar um tratamento adequado desses dados faltantes. Esse processo incluiu a identificação e a aplicação de técnicas de imputação para preencher as lacunas, além da verificação da integridade dos dados.

    Abaixo, você pode conferir o código utilizado para realizar o tratamento dos dados:
    """)

    # Código para exibição
    code = """
    import pandas as pd

    # Carregar o DataFrame
    renda = pd.read_csv(r"C:\\Users\\g_abi\\OneDrive\\Documentos\\Data Science\\Data Science EBAC\\Módulo 16\\Projeto\\projeto 2\\input\\previsao_de_renda.csv")

    # Tratamento dos dados
    renda = (renda.assign(data_ref=lambda x: pd.to_datetime(x.data_ref))
            .pipe(lambda x: x.drop_duplicates(subset='id_cliente', keep='first'))
            .assign(tempo_emprego=lambda x: x.tempo_emprego.fillna(0))
            .pipe(lambda x: x.drop(columns='Unnamed: 0')))
    """

    st.code(code, language='python')
    
    st.subheader('Data Frame')
    st.dataframe(renda)

elif selected_page == 'ANÁLISE DAS VÁRIAVEIS CATEGÓRICAS':

    st.subheader('ANÁLISE DAS VARIÁVEIS CATEGÓRICAS' , divider = 'blue')
    st.write('Nesta etapa, criei uma nova coluna no DataFrame chamada classe_renda. A renda foi dividida em 5 classes, conforme os valores propostos pelo IBGE, permitindo uma avaliação mais precisa da influência das variáveis categóricas sobre cada classe de renda.')
    
    df = renda.groupby('classe_renda')[['sexo']].value_counts().reset_index()
    df2 =renda.groupby('classe_renda')[['tipo_renda']].value_counts().reset_index()
    df3 = renda.groupby('classe_renda')['educacao'].value_counts().reset_index()
    df4 = renda.groupby('classe_renda')['estado_civil'].value_counts().reset_index()

    fig , ax = plt.subplots( 4 , 1 , figsize = (12 , 50))

    sns.barplot(x='classe_renda', y='count', hue='sexo', data=df , ax = ax[0])
    ax[0].set_title('Distribuição de Sexo por Classe de Renda')
    ax[0].tick_params(axis='x', rotation=10)

    sns.barplot(x='classe_renda', y='count', hue='tipo_renda', data=df2 , ax = ax[1])
    ax[1].set_title('Tipo de renda por Classe de Renda')
    ax[1].tick_params(axis='x', rotation=10)

    sns.barplot(x='classe_renda', y='count', hue='educacao', data=df3 , ax = ax[2])
    ax[2].set_title('Nível educacional por Classe de renda')
    ax[2].tick_params(axis='x', rotation=10)

    sns.barplot(x='classe_renda', y='count', hue='estado_civil', data=df4 , ax = ax[3])
    ax[3].set_title('Estado civíl por Classe de Renda')
    ax[3].tick_params(axis='x', rotation=10)

    plt.subplots_adjust(hspace=0.2)
    st.pyplot(plt)

elif selected_page == 'CORRELAÇÃO ENTRE VARIÁVEIS QUANTITATIVAS':
    st.subheader('CORRELAÇÃO ENTRE VARIÁVEIS QUANTITATIVAS' , divider= 'blue')
    st.write('Usamos este gráfico para visualizar as correlações com a váriavel renda')

    renda_corr = (renda.drop(columns= 'id_cliente').select_dtypes(include= ['int64' , 'float64']).corr())

    cross_tab = pd.crosstab( renda.posse_de_veiculo , renda.renda)

    plt.figure(figsize=(12 , 10))
    sns.heatmap(renda_corr , linewidths= 1 , annot= True , cmap= 'Blues' , center= 0)
    st.pyplot(plt)

    fig, ax = plt.subplots(4 , 1 , figsize = (12 , 50))

    sns.scatterplot(x = 'tempo_emprego' , y = 'renda' , data = renda , ax = ax[0])
    ax[0].set_title('Renda x Tempo de emprego');

    sns.scatterplot(x = 'idade' , y = 'renda' , data = renda , ax = ax[1])
    ax[1].set_title('Renda x Idade');

    sns.barplot(x = 'qt_pessoas_residencia' , y = 'renda' , data = renda , ax = ax[2])
    ax[2].set_title('Renda x Qtd pessoas na residencia');

    sns.barplot(x = 'qtd_filhos' , y = 'renda' , data = renda , ax = ax[3])
    ax[3].set_title('Renda x Qtd de filhos');

    st.pyplot(plt)

elif selected_page == 'ANÁLISES TEMPORAIS':

    st.subheader('ANÁLISES TEMPORAIS' , divider='blue')
    st.write('Nesta etapa, busquei analisar a variação da renda em decorrencia de outras váriaveis ao decorrer do tempo')

    plt.figure(figsize= (25 , 10))
    sns.lineplot(x = 'data_ref' , y = 'renda' , hue = 'sexo'  , data = renda)
    plt.title('Renda por sexo ao Longo do Tempo')

    st.pyplot(plt)

    plt.figure(figsize=(25 , 10))
    sns.lineplot(x = 'data_ref' , y = 'renda' , hue = 'educacao' , data = renda , errorbar = ('ci' , 0))
    plt.title('Renda por nível de educacional ao Longo do Tempo')

    st.pyplot(plt)

    plt.figure(figsize=(25 , 10))
    sns.lineplot(x = 'data_ref' , y = 'renda' , hue = 'tipo_residencia' , data = renda , errorbar = ('ci' , 0))
    plt.title('Renda por tipo de residência ao Longo do Tempo')

    st.pyplot(plt)

    plt.figure(figsize=(25 , 10))
    sns.lineplot(x = 'data_ref' , y = 'renda' , hue = 'estado_civil' , data = renda , errorbar = ('ci' , 0))
    plt.title('Renda por estado civíl ao Longo do Tempo')

    st.pyplot(plt)





