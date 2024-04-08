# Flusso per la modifica dopo meeting flag 13/02/2024
# ---------------------------------------------------
# Dopo input dati seleziono le piattaforme
# Funzione per il confronto su codici e quantità che mi restituisce l'elenco delle piattaforme non omogenee | stampa a video del dataframe di tutte le piattaforme con flaggate quelle non ok
# Colonna flaggabile per andare come input alla funzione successiva
# Funzione per visualizzare gli alberi apppaiati (già esistente)
# L'elenco serve poi per elimminare le piattaforme | for piattaforma in elenco --> cerco il codice nella distinta --> flag di tutti i suoi figli con Elimina = 1
# poi filtro le distinte
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import networkx as nx
#from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from PIL import Image # serve per l'immagine?
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
st.set_option('deprecation.showPyplotGlobalUse', False)
from io import BytesIO
#from pyxlsb import open_workbook as open_xlsb
#import xlsxwriter
import io
from io import StringIO
#from fuzzywuzzy import fuzz

from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

st.set_page_config(layout="wide")

tab0, tab1, tab2, tab3, tab4 = st.tabs(['Input dati','Analisi piattaforme','Elaborazione dati','Action Item liv2', 'Delta Qty' ])

with tab0: #Input dei dati
    # Layout app
    url_immagine='https://github.com/MarcelloGalimberti/Delta_BOM/blob/main/Ducati-Multistrada-V4-2021-008.jpeg?raw=true'
    st.title('Delta BOM D33 vs ZD33')
    st.subheader('02 aprile 2024', divider ='red')

# Importazione file da Streamlit

    uploaded_SAP = st.file_uploader("Carica la distinta D33")
    if not uploaded_SAP:
        st.stop()
    SAP_cap = pd.read_excel(uploaded_SAP)

    st.write(SAP_cap[0:2])

    uploaded_PLM = st.file_uploader("Carica la distinta ZD33")
    if not uploaded_PLM:
        st.stop()
    PLM = pd.read_excel(uploaded_PLM)

    st.write(PLM[0:2])

    sku = PLM['Numero componenti'].iloc[0]
    st.sidebar.header(f'SKU: :red[{sku}]')
    sku_moto = SAP_cap['Materiale'].iloc[0]

    anagrafica_sap = SAP_cap[['Materiale','Testo breve oggetto']]
    anagrafica_sap.drop_duplicates(inplace=True)
    anagrafica_plm = PLM[['Numero componenti','Testo breve oggetto']]
    anagrafica_plm.drop_duplicates(inplace=True)

# Preparazione distinte non filtrate
    def sap_raw (df):
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        df = df[['Liv.','Materiale','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Tecn.','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Ril.Prod.','Ril.Ric.','Testo posizione riga 1',
        'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]
        df.rename(columns={'Materiale':'Articolo','Qtà comp. (UMC)':'Qty'},inplace=True)
        return df

    def plm_raw (df):
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        #df['Liv.']=df['Liv.'].astype(int)-1
        df = df[['Liv.','Numero componenti','Qtà comp. (UMC)','Merce sfusa','Ril. progettazione','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Rilevante produzione','Cd.parte di ricambio','Testo posizione riga 1',
        'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]
        df.rename(columns={'Numero componenti':'Articolo','Qtà comp. (UMC)':'Qty','Merce sfusa':'MerceSfusa (BOM)','Ril. progettazione':'Ril.Tecn.','Rilevante produzione':'Ril.Prod.','Cd.parte di ricambio':'Ril.Ric.'},
                inplace=True)
        #df = df.fillna(0) eliminato 28/12
        df['Liv.']= df['Liv.'].astype(int)
        df['MerceSfusa (BOM)']=df['MerceSfusa (BOM)'].apply(lambda x: 'Sì' if x == 'X' else 'No' )        
        df['Ril.Tecn.']=df['Ril.Tecn.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Prod.']=df['Ril.Prod.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Ric.']=df['Ril.Ric.'].apply(lambda x: True if x  =='X' else False)      
        return df

    SAP_raw = sap_raw(SAP_cap)
    PLM_raw = plm_raw(PLM)

with tab1:# Ricerca piattaforme
    colA, colB, colC = st.columns([1,1,1])

    codici_piattaforma = ['P','S','T','BR','BRA']
    codici_piattaforma = ['P','S','T','BR']
    piattaforme_SAP = list(set(list(SAP_raw[[any(digit in articolo[3:5] for digit in codici_piattaforma)for articolo in SAP_raw.Articolo.astype(str)]].Articolo)))
    piattaforme_PLM = list(set(list(PLM_raw[[any(digit in articolo[3:5] for digit in codici_piattaforma)for articolo in PLM_raw.Articolo.astype(str)]].Articolo)))

    with colA:
        st.subheader('Count piattaforme', divider='red')
        st.write('SAP: {} codici di piattaforma'.format(len(piattaforme_SAP)))
        st.write('PLM: {} codici di piattaforma'.format(len(piattaforme_PLM)))
        st.write('Righe totali distinta SAP',len(SAP_raw))
        st.write('Righe totali distinta PLM',len(PLM_raw))

    SAP_not_PLM = list(set(piattaforme_SAP).difference(piattaforme_PLM))
    PLM_not_SAP = list(set(piattaforme_PLM).difference(piattaforme_SAP))
     
    with colB:
        st.subheader('Piattaforme in SAP e non in PLM', divider='gray')
        for i in range(len(SAP_not_PLM)):
            st.write(SAP_not_PLM[i])
        
    with colC:
        st.subheader('Piattaforme in PLM e non in SAP', divider ='grey')
        for i in range(len(PLM_not_SAP)):
            st.write(PLM_not_SAP[i])

    piattaforme = list(set(piattaforme_SAP)&set(piattaforme_PLM))
    piattaforme_frame = pd.DataFrame(piattaforme,)
    piattaforme_frame = piattaforme_frame.merge(anagrafica_sap, how='left', left_on = 0,right_on='Materiale')
  
    piattaforme_frame['Check_presenza']= None
    piattaforme_frame['Check_qty']= None

# Test omogeneità
    
    presenza  = {}
    qty = {}
    
    for sku in piattaforme:
        for i in range(len(SAP_raw)):
            articolo = SAP_raw.Articolo.iloc[i]
            if articolo == sku:
                albero_sap = SAP_raw[['Liv.','Articolo','Testo breve oggetto','Qty']].copy()
                albero_sap['Piattaforma']= None
                albero_sap.Piattaforma.iloc[i]=1
                livello = albero_sap['Liv.'].iloc[i]
                j=1
                if (j+1) == len(albero_sap):
                    break
                while albero_sap['Liv.'].iloc[j+i]>livello:
                    albero_sap.Piattaforma.iloc[j+i]=1
                    j+=1
                    if j+1 == len(albero_sap):
                        break
                albero_sap = albero_sap[albero_sap.Piattaforma == 1]
                albero_sap.drop('Piattaforma', axis=1, inplace=True)

        for k in range(len(PLM_raw)):
            n=1
            articolo = PLM_raw.Articolo.iloc[k]
            if articolo == sku:
                albero_plm = PLM_raw[['Liv.','Articolo','Testo breve oggetto','Qty']].copy()
                albero_plm['Piattaforma']= None
                albero_plm.Piattaforma.iloc[k]=1
                livello = albero_plm['Liv.'].iloc[k]
                n=1
                if (k+n+1) == len(albero_plm):
                    break
                while albero_plm['Liv.'].iloc[n+k]>livello:
                    albero_plm.Piattaforma.iloc[n+k]=1
                    n+=1
                    if k+n+1 == len(albero_plm):
                        break
                albero_plm = albero_plm[albero_plm.Piattaforma == 1]
                albero_plm.drop('Piattaforma', axis=1, inplace=True)    
    
        #confronto
        codici_sap = set(list(albero_sap.Articolo))
        codici_plm = set(list(albero_plm.Articolo))
        no_plm = list(codici_sap.difference(codici_plm))
        no_sap = list(codici_plm.difference(codici_sap))
        if no_plm==[] and no_sap==[]:
            pres='Ok'
            presenza[sku]=pres
        else:
            pres='Differenza di codici'
            presenza[sku]='Differenze di codici'

        if pres=='Ok':
            sap_sum = albero_sap.groupby(by=['Liv.','Articolo','Testo breve oggetto']).sum()
            plm_sum = albero_plm.groupby(by=['Liv.','Articolo','Testo breve oggetto']).sum()
            sap_sum = sap_sum.merge(plm_sum, how='left',left_on='Articolo',right_on='Articolo')
            sap_sum['delta'] = sap_sum['Qty_x']-sap_sum['Qty_y']
            check = sap_sum.delta.sum()
            if check == 0:
                qty[sku]='Ok'
            else:
                qty[sku]='Delta quantità'
    
    for i in range(len(piattaforme_frame)):
        codice = piattaforme_frame.Materiale.iloc[i]
        piattaforme_frame['Check_presenza'].iloc[i]=presenza[codice]
        if presenza[codice]=='Ok':    
            piattaforme_frame['Check_qty'].iloc[i]=qty[codice]

    st.subheader(' ')
    st.subheader(' ')
    colD, colE = st.columns([1,1])
    with colD:
        st.subheader('Piattaforme omogenee', divider='red')
        piattaforme_ok = piattaforme_frame[piattaforme_frame.Check_qty.astype(str)=='Ok']  
        st.dataframe(piattaforme_ok.drop(columns=0), width=2000)
        st.write('{} piattaforme omogenee'.format(len(piattaforme_ok)))

    with colE:
        st.subheader('Piattaforme con delta', divider='red')
        piattaforme_delta = piattaforme_frame[piattaforme_frame.Check_qty.astype(str)!='Ok']
        st.dataframe(piattaforme_delta.drop(columns=0), width=2000)
        st.write('{} piattaforme con delta'.format(len(piattaforme_delta)))
        
    st.subheader(' ')
    st.subheader(' ')
    st.subheader('Confronto alberi', divider ='red')

    def estrai(sku, distinta):
        for i in range(len(distinta)):
            articolo = distinta.Articolo.iloc[i]
            if articolo == sku:
                albero = distinta[['Liv.','Articolo','Testo breve oggetto','Qty']].copy()
                albero['Piattaforma']= None
                albero.Piattaforma.iloc[i]=1
                livello = albero['Liv.'].iloc[i]
                j=1
                if (j+i+1) == len(albero):
                    break
                while albero['Liv.'].iloc[j+i]>livello:
                    albero.Piattaforma.iloc[j+i]=1
                    j+=1
                    if j+i+1 == len(albero):
                        break
        albero = albero[albero.Piattaforma == 1]
        albero.drop('Piattaforma', axis=1, inplace=True)
        albero = albero.reset_index(drop=True)
        return albero

    def add_padre(df):
        # la funzione prende in ingresso un albero (df) di distinta e accanto a ogni codice scrive il nome di suo padre
        df['Padre']=None
        for i in range(1, len(df)):
            livello = df['Liv.'].iloc[i]
            j=1
            while df['Liv.'].iloc[i-j] >= livello:
                j+=1
            else:
                padre = df.Articolo.iloc[i-j]
                df.Padre.iloc[i]=padre
                continue
        df.Padre.iloc[0] = 'Nessuno'
        df['key']=df.Articolo + ' figlio di ' + df.Padre
        return df

    def compara_strutture_piattaforme(albero_SAP, albero_PLM):
        
        albero_SAP = albero_SAP.rename(columns={'Liv.':'livello','Articolo':'articolo','Testo breve oggetto':'descrizione','Qty':'Qty SAP'})
        albero_SAP = albero_SAP[['livello','articolo','descrizione','Qty SAP','key']] # mantenute qty
        albero_PLM = albero_PLM.rename(columns={'Liv.':'livello','Articolo':'articolo','Testo breve oggetto':'descrizione','Qty':'Qty PLM'})
        albero_PLM = albero_PLM[['livello','articolo','descrizione','Qty PLM','key']] # mantenute qty   
        
        albero_PLM['Liv'] = [int(stringa[-1]) for stringa in albero_PLM['livello'].astype(str)]
        albero_PLM=albero_PLM.drop('livello',axis=1)
        albero_SAP['Liv'] = [int(stringa[-1]) for stringa in albero_SAP['livello'].astype(str)]
        albero_SAP = albero_SAP.drop('livello',axis=1)

        albero_PLM_sum = albero_PLM[['key','Qty PLM']].groupby(by='key').sum()
        albero_PLM_sum = albero_PLM_sum.rename(columns={'Qty PLM':'Totale_PLM'})
        albero_SAP_sum = albero_SAP[['key','Qty SAP']].groupby(by='key').sum()
        albero_SAP_sum = albero_SAP_sum.rename(columns={'Qty SAP':'Totale_SAP'})

        albero_SAP = albero_SAP.merge(albero_SAP_sum, how='left',left_on='key',right_on='key')
        albero_PLM = albero_PLM.merge(albero_PLM_sum, how='left',left_on='key',right_on='key')
  
        #sap_new = albero_SAP.merge(albero_PLM, how='left',left_on='articolo',right_on='articolo')
        sap_new = albero_SAP.merge(albero_PLM, how='left',left_on='key',right_on='key')

        sap_new['riportato'] = np.where(sap_new.descrizione_y.astype(str)=='nan','si',None)

        albero_PLM['new_index']=None
        for i in range(len(albero_PLM)):
            albero_PLM['new_index'].iloc[i]=100*i    
            
        sap_new['padre_plm']=None
        for i in range(len(sap_new)):
            n=0
            if sap_new.riportato.iloc[i]=='si':
                for j in range(20):
                    if sap_new.riportato.astype(str).iloc[i-j] == 'None':               
                        ##print(sap_new.riportato.astype(str).iloc[i-j])
                        #codice = sap_new.articolo.iloc[i-j] 
                        key = sap_new.key.iloc[i-j]

                        #indice_plm = albero_PLM[albero_PLM.articolo == codice].index[0]
                        indice_plm = albero_PLM[albero_PLM.key == key].index[0]


                        sap_new.padre_plm.iloc[i] = indice_plm*100 +1
                        sap_new['Liv_x'].iloc[i]=None
                        break

        to_append = sap_new[['articolo_y','descrizione_y','Liv_x','padre_plm','Qty SAP','key']][sap_new.riportato=='si']
        to_append = to_append.rename(columns={'descrizione_y':'descrizione','Liv_x':'Liv','padre_plm':'new_index'})
        albero_PLM = pd.concat([albero_PLM,to_append])  
        albero_PLM = albero_PLM.sort_values(by='new_index')
        albero_PLM = albero_PLM.drop(columns=['articolo_y','Qty SAP'])
        plm_new = albero_PLM.merge(albero_SAP, how='left',left_on='key',right_on='key')
        plm_new['Articolo'] = np.where(plm_new.articolo_x.astype(str) != 'nan', plm_new.articolo_x, plm_new.articolo_y)
    
        plm_new = plm_new.rename(columns={'descrizione_x':'descrizione_PLM','Liv_x':'Liv_PLM','descrizione_y':'descrizione_SAP','Liv_y':'Liv_SAP','Qty SAP_y':'Qty SAP','Totale_SAP_x':'Totale_SAP'})
        plm_new.drop(columns=['new_index','key','articolo_x','articolo_y'], axis=1, inplace=True)
        plm_new = plm_new[['Articolo','descrizione_PLM','Qty PLM','Liv_PLM','descrizione_SAP','Qty SAP','Liv_SAP','Totale_PLM','Totale_SAP']]
        return plm_new
    
    def color_mancante(val):
        color = 'grey' if str(val)=='nan' or str(val)=='<NA>' else 'black'
        return f'background-color: {color}'
    def color_mancante2(val):
        color = 'red' if val !=0  else 'black'
        return f'background-color: {color}'

    codice_piattaforma = st.text_input('Inserire codice da confrontare')
    #if not codice_piattaforma:
    #    st.stop()
    if codice_piattaforma:    
        piattaforma_SAP = estrai(codice_piattaforma, SAP_raw)
        piattaforma_PLM = estrai(codice_piattaforma, PLM_raw)

        piattaforma_SAP_2 = add_padre(piattaforma_SAP)
        piattaforma_PLM_2 = add_padre(piattaforma_PLM)

        piattaforme_compare = compara_strutture_piattaforme(piattaforma_SAP_2,piattaforma_PLM_2)
        
        piattaforme_compare['Liv_PLM'] = piattaforme_compare['Liv_PLM'].astype('Int64')
        piattaforme_compare['Liv_SAP'] = piattaforme_compare['Liv_SAP'].astype('Int64')
        piattaforme_compare['Delta_Qty'] = piattaforme_compare['Qty PLM'] - piattaforme_compare['Qty SAP']
        piattaforme_compare['Delta_Tot'] = piattaforme_compare['Totale_PLM'] - piattaforme_compare['Totale_SAP']
        
        piattaforme_compare = piattaforme_compare[((piattaforme_compare.Delta_Tot == 0) & (piattaforme_compare.Delta_Qty != 0))==False]
        #piattaforme_compare.drop_duplicates(inplace=True)

        piattaforme_compare['Indentato'] = [(livello-2)* 10 * ' ' if str(livello) != '<NA>' else None for livello in piattaforme_compare['Liv_PLM']]
        piattaforme_compare['descrizione_PLM'] = piattaforme_compare.Indentato + piattaforme_compare.descrizione_PLM
        piattaforme_compare['Indentato_SAP'] = [(livello-2)* 10 * ' ' if str(livello) != '<NA>' else None for livello in piattaforme_compare['Liv_SAP']]
        piattaforme_compare['descrizione_SAP'] = piattaforme_compare.Indentato_SAP + piattaforme_compare.descrizione_SAP
        st.dataframe(piattaforme_compare.drop(columns=['Indentato','Indentato_SAP'], axis=1).style.applymap(color_mancante, subset=['descrizione_PLM','descrizione_SAP','Liv_PLM','Liv_SAP','Qty SAP','Qty PLM']), width=2500)

        st.write('Non in PLM:', len(piattaforme_compare[piattaforme_compare.descrizione_PLM.astype(str) == 'nan']))
        st.write('Non in SAP:', len(piattaforme_compare[piattaforme_compare.descrizione_SAP.astype(str) == 'nan']))
        deltaqty=piattaforme_compare[piattaforme_compare.Delta_Qty != 0].drop(columns=['Indentato','Indentato_SAP'],axis=1)
        st.write('Correggere qty:',len(deltaqty[deltaqty.Delta_Qty.astype(str)!='nan']))

        st.write('Righe con ∆ QTY')
        st.divider()
        
        st.dataframe(deltaqty, width=2500)


    colonne_piatt = ['Articolo', 'descrizione_PLM','Qty PLM','Liv_PLM','descrizione_SAP','Qty SAP','Liv_SAP','Totale_PLM','Totale_SAP','Delta_Qty','Delta_Tot','Indentato','Indentato_SAP']

    #colonne_piatt = piattaforme_compare.columns
    complessivo = pd.DataFrame(columns=colonne_piatt)
    complessivo['Piattaforma']=None
    complessivo['SKU']=None

    for codice_piattaforma in piattaforme_frame['Materiale']:
        piattaforma_SAP = estrai(codice_piattaforma, SAP_raw)
        piattaforma_PLM = estrai(codice_piattaforma, PLM_raw)

        piattaforma_SAP_2 = add_padre(piattaforma_SAP)
        piattaforma_PLM_2 = add_padre(piattaforma_PLM)

        piattaforme_compare = compara_strutture_piattaforme(piattaforma_SAP_2,piattaforma_PLM_2)
        
        piattaforme_compare['Liv_PLM'] = piattaforme_compare['Liv_PLM'].astype('Int64')
        piattaforme_compare['Liv_SAP'] = piattaforme_compare['Liv_SAP'].astype('Int64')
        piattaforme_compare['Delta_Qty'] = piattaforme_compare['Qty PLM'] - piattaforme_compare['Qty SAP']
        piattaforme_compare['Delta_Tot'] = piattaforme_compare['Totale_PLM'] - piattaforme_compare['Totale_SAP']
        piattaforme_compare = piattaforme_compare[((piattaforme_compare.Delta_Tot == 0) & (piattaforme_compare.Delta_Qty != 0))==False]
        piattaforme_compare['Piattaforma']=codice_piattaforma
        piattaforme_compare['SKU']=sku_moto

        complessivo = pd.concat([complessivo, piattaforme_compare])

    st.subheader('Complessivo codici piattaforma co ∆ (esclusi codici non in SAP)')
    st.write('Quantità codici con ∆')
    st.write(len(complessivo[(complessivo.Delta_Qty != 0) & (complessivo.descrizione_SAP.astype(str)!= 'nan')]))
    st.dataframe(complessivo[(complessivo.Delta_Qty != 0) & (complessivo.descrizione_SAP.astype(str)!= 'nan')].drop(columns=['Indentato','Indentato_SAP'],axis=1))
    file_download = complessivo[(complessivo.Delta_Qty != 0) & (complessivo.descrizione_SAP.astype(str)!= 'nan')].drop(columns=['Indentato','Indentato_SAP'],axis=1)


    nome_file = sku_moto +' - '+ 'livelli > 2'
    #if st.button('Scarica Excel'):
    #    file_download.to_excel('/Users/Alessandro/Documents/AB/Clienti/ADI!/Ducati/DeltaBOM/Download_deltabom/'+nome_file+'.xlsx')
    
    #st.dataframe(piattaforme_compare[piattaforme_compare.Delta_Qty != 0].drop(columns=['Indentato','Indentato_SAP'],axis=1), width=2500)

   # st.stop()   
# Aggiornamento dei filtri
    
with tab2: # elaborazione livelli e gruppi tecnici
    def importa_cap_con_descrizone_D33_old (df):
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        df = df[['Liv.','Materiale','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Tecn.','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Ril.Prod.','Ril.Ric.','Testo posizione riga 1',
                'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]
        df.rename(columns={'Materiale':'Articolo','Qtà comp. (UMC)':'Qty'},inplace=True)
        #df = df.fillna(0) eliminato 28/12
        df['Liv.']= df['Liv.'].astype(int)
        
        # step 1: eliminare tutti i figli con padre livello 2 merce sfusa
        df['Eliminare'] = 0
        for i in range(len(df)):
            if i == len(df):
                break
            if (df.loc[i,'MerceSfusa (BOM)'] == 'Sì' and df.loc[i,'Liv.']== 2):
                livello_padre = df.loc[i,'Liv.']
                #
                if ((df.loc[i,'Ril.Tecn.']==False) and (df.loc[i,'Ril.Prod.']==False) and (df.loc[i,'Ril.Ric.']==False)):
                    df.loc[i,'Eliminare'] = 1 #era 0
                else:
                    df.loc[i,'Eliminare'] = 1
                #
                j = i
                if (j+1) == len(df):
                    break
                while df.loc[j+1,'Liv.']>livello_padre:
                    df.at[j+1,'Eliminare']=1
                    j+=1
                    if (j+1) == len(df):
                        break  
        df = df.loc[df['Eliminare']==0] 
    
        # step 2: eliminare il resto in base a rilevanza tecnica
        df = df.loc[
            (df['MerceSfusa (BOM)']=='No') ]
           # ((df['Ril.Prod.']==True)&(df['MerceSfusa (BOM)']=='No')) |
           # ((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) |
          #  (((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)] 
        
        df = df.drop(columns=['MerceSfusa (BOM)','Ril.Tecn.','Eliminare','Ril.Prod.','Ril.Ric.']) 
        df.reset_index(drop=True, inplace=True) 
        return df

    def importa_cap_con_descrizone_D33 (df): # in uso dopo introduzione analisi preliminare piattaforme
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        df = df[['Liv.','Materiale','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Tecn.','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Ril.Prod.','Ril.Ric.','Testo posizione riga 1',
                'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]
        df.rename(columns={'Materiale':'Articolo','Qtà comp. (UMC)':'Qty'},inplace=True)
        df['Liv.']= df['Liv.'].astype(int)
        
        # step 1: eliminare tutti i figli con padre livello 2 merce sfusa
        df['Eliminare'] = 0
        for i in range(len(df)):
            if i == len(df):
                break
            if (df.loc[i,'MerceSfusa (BOM)'] == 'Sì'): #and df.loc[i,'Liv.']== 2):
                df.loc[i,'Eliminare'] = 1
                livello_padre = df.loc[i,'Liv.']
    
                j = i
                if (j+1) == len(df):
                    break
                while df.loc[j+1,'Liv.']>livello_padre:
                    df.at[j+1,'Eliminare']=1
                    j+=1
                    if (j+1) == len(df):
                        break  
        df = df.loc[df['Eliminare']==0] 
        
        df = df.drop(columns=['MerceSfusa (BOM)','Ril.Tecn.','Eliminare','Ril.Prod.','Ril.Ric.']) 
        df.reset_index(drop=True, inplace=True) 
        return df

    def importa_plm_con_descrizone_DAA_old (df):
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        df = df[['Liv.','Numero componenti','Qtà comp. (UMC)','Merce sfusa','Ril. progettazione','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Rilevante produzione','Cd.parte di ricambio','Testo posizione riga 1',
                'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]

        df.rename(columns={'Numero componenti':'Articolo','Qtà comp. (UMC)':'Qty','Merce sfusa':'MerceSfusa (BOM)','Ril. progettazione':'Ril.Tecn.','Rilevante produzione':'Ril.Prod.','Cd.parte di ricambio':'Ril.Ric.'},
                inplace=True)
        #df = df.fillna(0) eliminato 28/12
        df['Liv.']= df['Liv.'].astype(int)

        df['MerceSfusa (BOM)']=df['MerceSfusa (BOM)'].apply(lambda x: 'Sì' if x == 'X' else 'No' )
        
        df['Ril.Tecn.']=df['Ril.Tecn.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Prod.']=df['Ril.Prod.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Ric.']=df['Ril.Ric.'].apply(lambda x: True if x  =='X' else False)

        
        #correzione gruppi teecnici mancanti
        for i in range(len(df)): 
            if (df['Liv.'].iloc[i] >2) and (df['Gruppo Tecnico'].astype(str).iloc[i]=='nan'):
                df['Gruppo Tecnico'].iloc[i] = df['Gruppo Tecnico'].iloc[i-1]
                df['Descr. Gruppo Tecnico'].iloc[i] = df['Descr. Gruppo Tecnico'].iloc[i-1]

        # step 1: eliminare tutti i figli con padre livello 2 merce sfusa
        df['Eliminare'] = 0
        for i in range(len(df)):
            if i == len(df):
                break
            if (df.loc[i,'MerceSfusa (BOM)'] == 'Sì' and df.loc[i,'Liv.']== 2):
                livello_padre = df.loc[i,'Liv.']
                #
                #if ((df.loc[i,'Ril.Tecn.']==False) and (df.loc[i,'Ril.Prod.']==False) and (df.loc[i,'Ril.Ric.']==False)) or ((df.loc[i,'Ril.Prod.']==True) and (df.loc[i,'Ril.Ric.']==True)) :
                if ((df.loc[i,'Ril.Tecn.']==False) and (df.loc[i,'Ril.Prod.']==False) and (df.loc[i,'Ril.Ric.']==False)):
                    df.loc[i,'Eliminare'] = 1 # era 0 comunque questa configurazione non esiste nel PLM
                else:
                    df.loc[i,'Eliminare'] = 1
                #
                j = i
                if (j+1) == len(df):
                    break
                while df.loc[j+1,'Liv.']>livello_padre:
                    df.at[j+1,'Eliminare']=1
                    j+=1
                    if (j+1) == len(df):
                        break  

            if (df.loc[i,'MerceSfusa (BOM)'] == 'No'):# and df.loc[i,'Liv.']== 2) : # eliminazione codici SOLO RICAMBIO
                if ((df.loc[i,'Ril.Tecn.']==True) and (df.loc[i,'Ril.Prod.']==False) and (df.loc[i,'Ril.Ric.']==True)):                   
                    df.loc[i,'Eliminare'] = 1
                else:
                    df.loc[i,'Eliminare'] = 0

        df = df.loc[df['Eliminare']==0] 
    
        # step 2: eliminare il resto in base a rilevanza tecnica
        df = df.loc[(df['MerceSfusa (BOM)']=='No')] 
                    #((df['Ril.Tecn.']==True)&(df['MerceSfusa (BOM)']=='Sì')) | # non esiste in PLM
                    #(((df['MerceSfusa (BOM)']=='Sì')&(df['Ril.Tecn.']==False)&(df['Ril.Prod.']==False)&(df['Ril.Ric.']==False))==True)] # non esiste in PLM
        
        df = df.drop(columns=['MerceSfusa (BOM)','Ril.Tecn.','Eliminare','Ril.Prod.','Ril.Ric.']) 
        df.reset_index(drop=True, inplace=True) 
        return df

    def importa_plm_con_descrizone_DAA (df): # in uso dopo introduzione analisi preliminare piattaforme
        df['Liv.']=df['Liv. esplosione'].str.replace('.','')
        df = df[['Liv.','Numero componenti','Qtà comp. (UMC)','Merce sfusa','Ril. progettazione','Testo breve oggetto','Gruppo Tecnico','Descr. Gruppo Tecnico','Rilevante produzione','Cd.parte di ricambio','Testo posizione riga 1',
                'Testo posizione riga 2','STGR','Descrizione Sottogruppo','Gruppo appartenenza','Descr. Gruppo Appartenenza']]

        df.rename(columns={'Numero componenti':'Articolo','Qtà comp. (UMC)':'Qty','Merce sfusa':'MerceSfusa (BOM)','Ril. progettazione':'Ril.Tecn.','Rilevante produzione':'Ril.Prod.','Cd.parte di ricambio':'Ril.Ric.'},
                inplace=True)

        df['Liv.']= df['Liv.'].astype(int)
        df['MerceSfusa (BOM)']=df['MerceSfusa (BOM)'].apply(lambda x: 'Sì' if x == 'X' else 'No' )       
        df['Ril.Tecn.']=df['Ril.Tecn.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Prod.']=df['Ril.Prod.'].apply(lambda x: True if x  =='X' else False)
        df['Ril.Ric.']=df['Ril.Ric.'].apply(lambda x: True if x  =='X' else False)

        #correzione gruppi teecnici mancanti
        for i in range(len(df)): 
            if (df['Liv.'].iloc[i] >2) and (df['Gruppo Tecnico'].astype(str).iloc[i]=='nan'):
                df['Gruppo Tecnico'].iloc[i] = df['Gruppo Tecnico'].iloc[i-1]
                df['Descr. Gruppo Tecnico'].iloc[i] = df['Descr. Gruppo Tecnico'].iloc[i-1]

        # step 1: eliminare tutti i figli con padre merce sfusa
        df['Eliminare'] = 0
        for i in range(len(df)):
            if i == len(df):
                break
            if (df.loc[i,'MerceSfusa (BOM)'] == 'Sì'):
                livello_padre = df.loc[i,'Liv.']
                df.loc[i,'Eliminare'] = 1

                j = i
                if (j+1) == len(df):
                    break
                while df.loc[j+1,'Liv.']>livello_padre:
                    df.loc[j+1,'Eliminare']=1
                    j+=1
                    if (j+1) == len(df):
                        break  

            if (df.loc[i,'MerceSfusa (BOM)'] == 'No') and (df.loc[i, 'Eliminare']==0):
                if ((df.loc[i,'Ril.Tecn.']==True) and (df.loc[i,'Ril.Prod.']==False) and (df.loc[i,'Ril.Ric.']==True)):                   
                    df.loc[i,'Eliminare'] = 1
                else:
                    df.loc[i,'Eliminare'] = 0

        df = df.loc[df['Eliminare']==0] 
        df = df.drop(columns=['MerceSfusa (BOM)','Ril.Tecn.','Eliminare','Ril.Prod.','Ril.Ric.']) 
        df.reset_index(drop=True, inplace=True) 
        return df

    # Ottengo working file

    # modifica 20231228
    SAP_BOM = importa_cap_con_descrizone_D33(SAP_cap)
    PLM_BOM = importa_plm_con_descrizone_DAA (PLM)

    SAP_raw = sap_raw(SAP_cap)
    PLM_raw = plm_raw(PLM)

    # Lista livelli 1  - serve per sottoalberi M, V, X
    # ok 20231228
    SAP_livelli_1 = SAP_BOM[SAP_BOM['Liv.'] == 1]
    lista_SAP_livelli_1 = SAP_livelli_1.Articolo.to_list()


    PLM_livelli_1 = PLM_BOM[PLM_BOM['Liv.'] == 1]
    lista_PLM_livelli_1 = PLM_livelli_1.Articolo.to_list()


    # Script per estrarre il sottoalbero delle BOM a partire da un SKU
    # ok 20231228
    def partizione (SKU,BOM):#------------------------------------------------------------------------------------------------------------------------------------SEGNALIBRO
        indice = BOM.index[BOM['Articolo'] == SKU].tolist()
        idx=indice[0]
        livello_SKU = BOM.iloc[idx,0]
        j=idx+1
        if j == len(BOM):
            indice_target = idx
            df = BOM.iloc[idx:indice_target+1,:]
        elif BOM.iloc[j,0] <= livello_SKU:  
            indice_target = j
            df = BOM.iloc[idx:indice_target,:]
        else:
            while BOM.iloc[j,0] > livello_SKU:  
                if (j+1) == len(BOM):
                    indice_target = j+1
                    df = BOM.iloc[idx:indice_target+1,:]
                    break
                j+=1  
                indice_target = j
                df = BOM.iloc[idx:indice_target,:]
        return df

    # trovo riga che inizia con 0029
    # ok 20231228
    indice_motore = 0
    for i in range (len (PLM_BOM)):
        if PLM_BOM.loc[i,'Articolo'].startswith('0029'):
            indice_motore = i
        # st.write(indice_motore)
        # st.write(PLM_BOM.loc[i,'Articolo'])
        # st.write(PLM_BOM)


    if indice_motore > 0:    # eliminazione righe e assegnazione gt
        # trovo codice motore
        codice_motore = PLM_BOM.loc[indice_motore,'Articolo']

        # estraggo albero motore

        albero_motore = partizione(codice_motore,PLM_BOM)
        indice_inizio = indice_motore
        indice_fine = indice_motore + len(albero_motore)

        # elimino livelli 3 (oppure Item Type == Engine Functional Group) e faccio salire di 1 tutti gli altri tranne il livello 2
        albero_motore['eliminare'] = False
        for i in range (len (albero_motore)):
            if albero_motore.loc[i+indice_inizio,'Articolo'] == codice_motore:
                pass
            elif albero_motore.loc[i+indice_inizio,'Liv.'] == 3:
                albero_motore.loc[i+indice_inizio,'eliminare'] = True
            else:
                albero_motore.loc[i+indice_inizio,'Liv.'] = albero_motore.loc[i+indice_inizio,'Liv.']-1

        # assegno i gruppi tecnici al motore
        for i in range(1,len(albero_motore)):
            if albero_motore.eliminare.iloc[i]==True:
                gruppo_tecnico = albero_motore.Articolo.astype(str).iloc[i][:2]+'.'
                des_gruppo = albero_motore['Testo breve oggetto'].iloc[i]
            albero_motore['Gruppo Tecnico'].iloc[i]=gruppo_tecnico
            albero_motore['Descr. Gruppo Tecnico'].iloc[i]=des_gruppo
                
        albero_motore_processato = albero_motore[albero_motore['eliminare']==False]
        albero_motore_processato.drop(columns=['eliminare'], inplace=True)
        albero_motore_processato.reset_index(inplace=True, drop=True)

        # slice PLM_BOM
        slice_1 = PLM_BOM.iloc[0:indice_inizio,]
        slice_3 = PLM_BOM.iloc[indice_fine:,]


        # aggiungere albero motore processato
        # questa plm bom valida solo se c'è motore, altrimenti 
        PLM_BOM = pd.concat([slice_1,albero_motore_processato,slice_3], ignore_index=True) # questo non va bene, perchè accodato in fondo è su X

    #**********ATTENZIONE********** SOLO PER CONFRONTO: SUI LIVELLI 2 DEL PLM SENZA GT (PERCHè GRUPPI MANUFACTURING) PRENDO GT DA SAP

    db_gruppi_tecnici_SAP = SAP_BOM[SAP_BOM['Liv.']==2]
    db_gruppi_tecnici_SAP = db_gruppi_tecnici_SAP[['Articolo','Gruppo Tecnico','Descr. Gruppo Tecnico']]
    db_gruppi_tecnici_SAP = db_gruppi_tecnici_SAP.drop_duplicates()

    #prendo i livelli 2 senza gruppo tecnico
    no_gt = list(PLM_BOM[(PLM_BOM['Gruppo Tecnico'].astype(str)=='nan') & (PLM_BOM['Liv.']==2)].Articolo)

    #filtro db_gruppi_tecnici_SAP solo su questi codici, così quando andrò a fare il merge non mi duplica righe
    db_gruppi_tecnici_SAP = db_gruppi_tecnici_SAP[[any(part in codice for part in no_gt) for codice in db_gruppi_tecnici_SAP.Articolo.astype(str)]]
    db_gruppi_tecnici_SAP = db_gruppi_tecnici_SAP.rename(columns={'Gruppo Tecnico':'GT_appoggio','Descr. Gruppo Tecnico':'DGT_appoggio'})

    st.write('Check PLM pre merge descrizione gruppi tecnici livello 2',len(PLM_BOM))
    PLM_BOM['Gruppo Tecnico']=[stringa[:2] for stringa in PLM_BOM['Gruppo Tecnico'].astype(str)] # qui fa diventare na il nan
    PLM_BOM = PLM_BOM.merge(db_gruppi_tecnici_SAP, how='left',left_on='Articolo',right_on='Articolo')

    st.write('Check PLM dopo merge descrizione gruppi tecnici livello 2',len(PLM_BOM))
    PLM_BOM['Gruppo Tecnico'] = np.where((PLM_BOM['Gruppo Tecnico'].astype(str)=='na')&(PLM_BOM['GT_appoggio'].astype(str)=='NO TITOLO'),PLM_BOM.GT_appoggio,PLM_BOM['Gruppo Tecnico'])
    PLM_BOM['Gruppo Tecnico'] = np.where(PLM_BOM['Gruppo Tecnico'].astype(str)=='na',[stringa[:2] for stringa in PLM_BOM.GT_appoggio.astype(str)],PLM_BOM['Gruppo Tecnico'])
    PLM_BOM['Descr. Gruppo Tecnico'] = np.where(PLM_BOM['Descr. Gruppo Tecnico'].astype(str)=='nan',PLM_BOM.DGT_appoggio,PLM_BOM['Descr. Gruppo Tecnico'])

    #correzione gruppi teecnici mancanti
    for i in range(len(PLM_BOM)): 
        if (PLM_BOM['Liv.'].iloc[i] >2) and (PLM_BOM['Gruppo Tecnico'].astype(str).iloc[i]=='na'):
            PLM_BOM['Gruppo Tecnico'].iloc[i] = PLM_BOM['Gruppo Tecnico'].iloc[i-1]
            PLM_BOM['Descr. Gruppo Tecnico'].iloc[i] = PLM_BOM['Descr. Gruppo Tecnico'].iloc[i-1]

    #scrivo errori di eredità gruppo tecnico
    st.subheader('Eccezioni regole Gruppi Tecnici DAA', divider='red')
    st.write('I codici in tabella non hanno potuto ereditare il gruppo tecnico del padre in quanto mancante')
    st.write(PLM_BOM[(PLM_BOM['Gruppo Tecnico'].astype(str)=='na') & (PLM_BOM['Liv.']>2)])


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # Ottengo moto M, V, X 20231228: attenzione, contengono la descrizione
    SAP_M = partizione(lista_SAP_livelli_1[0],SAP_BOM).reset_index(drop=True)
    PLM_M = partizione(lista_PLM_livelli_1[0],PLM_BOM).reset_index(drop=True)
    SAP_V = partizione(lista_SAP_livelli_1[1],SAP_BOM).reset_index(drop=True)
    PLM_V = partizione(lista_PLM_livelli_1[1],PLM_BOM).reset_index(drop=True)
    SAP_X = partizione(lista_SAP_livelli_1[2],SAP_BOM).reset_index(drop=True)
    PLM_X = partizione(lista_PLM_livelli_1[2],PLM_BOM).reset_index(drop=True)

    # 20231228 probabilmente questi df non serviranno
    SAP_BOM_con_descrizione = importa_cap_con_descrizone_D33(SAP_cap)

    SAP_M_descrizione = partizione(lista_SAP_livelli_1[0],SAP_BOM_con_descrizione).reset_index(drop=True) ####
    SAP_V_descrizione = partizione(lista_SAP_livelli_1[1],SAP_BOM_con_descrizione).reset_index(drop=True)
    SAP_X_descrizione = partizione(lista_SAP_livelli_1[2],SAP_BOM_con_descrizione).reset_index(drop=True)

    # Tabelle comparative
    # ok 20231228
    def tabella_comparativa (L_SAP,R_PLM):
        SAP_PVT = pd.pivot_table(L_SAP,
                            index='Liv.',
                            values = 'Articolo',
                            aggfunc=['count',pd.Series.nunique])
        SAP_PVT.rename(columns={'Articolo':'SAP','count':'numero righe','nunique':'codici'}
                    ,inplace=True)
        PLM_PVT = pd.pivot_table(R_PLM,
                            index='Liv.',
                            values = 'Articolo',
                            aggfunc=['count',pd.Series.nunique])
        PLM_PVT.rename(columns={'Articolo':'PLM','count':'numero righe','nunique':'codici'}
                    ,inplace=True)
        tabella = pd.concat([SAP_PVT,PLM_PVT],axis=1)
        tabella.fillna(0, inplace=True)
        tabella.sort_index(inplace=True)
        return tabella.astype(int)

    tabella = tabella_comparativa(SAP_BOM,PLM_BOM) # fare e visualizzare per moto D, M, V, X
    tabella.columns = ['_'.join(col).strip() for col in tabella.columns.values] 


    st.header('Tabella comparativa SAP - PLM livelli M, V, X', divider = 'red')
    col_a, col_b = st.columns([1,1])

    with col_a:
        st.subheader('Tabella', divider = 'red')
        st.write(tabella)

    with col_b:
        st.subheader('Descrizione', divider = 'red')
        st.write('Per ogni livello trovato in distinta:')
        st.write('**numero righe_SAP/PLM:** conteggio di codici, anche ripetuti, che compaiono in BOM')
        st.write('**codici_SAP/PLM:** conteggio di codici univoci che compaiono in BOM')   

with tab3: #Action Item livelli 2


    # Scegliere il livello da analizzare (M,V,X) in streamlit----------------------------------------------------------lo metto nella sidebar
        
    st.sidebar.header('Selezionare livello da analizzare', divider = 'red')
    livello_1 = st.sidebar.radio ('Scegli il tipo di distinta livello 1', ['M','V','X'], index=None)
    if not livello_1:
        st.stop()


    # aggiunto [['Liv.','Articolo','Qty']] 20231228
    if livello_1 == 'M':
        L_BOM_input = SAP_M[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
        R_BOM_input = PLM_M[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
    elif livello_1 == 'V':
        L_BOM_input = SAP_V[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
        R_BOM_input = PLM_V[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
    else:
        L_BOM_input = SAP_X[['Liv.','Articolo','Qty','Gruppo Tecnico']] 
        R_BOM_input = PLM_X[['Liv.','Articolo','Qty','Gruppo Tecnico']] 




    st.header(f'Analisi moto {livello_1}', divider = 'red')

    # 20231228 da verificare, forse non è mai utilizzata
    # Test confronto di un sottoalbero di un SKU
    # BOM_L: SAP_M | BOM_R: PLM_M | SKU tra ''
    def delta_SKU(SKU,L_BOM,R_BOM):
        df_L=partizione(SKU,L_BOM)
        df_R=partizione(SKU,R_BOM)
        compare = df_L.merge(df_R, how='outer', on='Articolo',
                                    indicator=True,
                                    left_index=False,right_index=False,
                                    suffixes = ('_SAP', '_PLM'))
        compare.rename(columns={'_merge':'Esito check codice'}, inplace=True)
        compare.fillna('',inplace=True)
        compare['Esito check codice'].replace(['both','right_only','left_only'],['Check ok','Non in SAP','Non in PLM'],
                                    inplace=True)
        return compare#, df_L, df_R

    # Comparazione livelli 2
    # 20231228 ok se alimentata dopo scelta ldel livello modificata
    def compara_livelli_2 (L_BOM,R_BOM): # L:SAP R:PLM
        SAP_codici_liv_2 = L_BOM[L_BOM['Liv.'] == 2]
        SAP_codici_liv_2.drop(columns=['Liv.','Qty'], inplace=True)
        SAP_codici_liv_2.drop_duplicates(inplace=True)
        PLM_codici_liv_2 = R_BOM[R_BOM['Liv.'] == 2]
        PLM_codici_liv_2.drop(columns=['Liv.','Qty'], inplace=True)
        PLM_codici_liv_2.drop_duplicates(inplace=True)
        comparelev_2 = SAP_codici_liv_2.merge(PLM_codici_liv_2, how='outer', on='Articolo',
                                    indicator=True,
                                    left_index=False,right_index=False,
                                    suffixes = ('_SAP', '_PLM'))
        comparelev_2.rename(columns={'_merge':'Esito check codice'}, inplace=True)
        comparelev_2['Esito check codice'].replace(['both','right_only','left_only'],['Check ok','Non in SAP','Non in PLM'],
                                    inplace=True)
        return comparelev_2 # capire se si può togliere

    comparelev_2 = compara_livelli_2 (L_BOM_input,R_BOM_input)
    comparelev_2.sort_values(by='Esito check codice', inplace=True, ascending=False) # Ok 348 [0]| Non in SAP 9 [2]| Non in PLM 22 [1]


    check_ok = len(comparelev_2[comparelev_2['Esito check codice'] == 'Check ok'])
    #st.write('check ok', check_ok)
    non_in_sap = len(comparelev_2[comparelev_2['Esito check codice'] == 'Non in SAP'])
    #st.write('non in sap', non_in_sap)
    non_in_plm = len(comparelev_2[comparelev_2['Esito check codice'] == 'Non in PLM'])
    #st.write('non in plm', non_in_plm)

    kpi_liv2 = (non_in_plm + non_in_sap)/(check_ok+non_in_sap+non_in_plm)*100
    lista_Venn=[check_ok,non_in_plm, non_in_sap]

    # Diagramma di Venn
    fig= plt.figure(figsize=(12,6))
    venn2(subsets =
        (lista_Venn[1],lista_Venn[2],lista_Venn[0]),
        set_labels=('SAP liv.2','PLM liv.2'),
        alpha=0.5,set_colors=('red', 'yellow'))

    df_confronto_lev2 = comparelev_2[comparelev_2['Esito check codice'] != 'Check ok']
    df_confronto_lev2[['Azione','Responsabile','Due date','Status']] = ""

    #st.write(df_confronto_lev2)
    #st.stop()

    # Aggiunto algo per descrizione e gruppo tecnico

    if livello_1 == 'M':
        df_SAP_descrizione = SAP_M_descrizione[['Articolo','Testo breve oggetto','Gruppo Tecnico']]
    elif livello_1 == 'V':
        df_SAP_descrizione = SAP_V_descrizione[['Articolo','Testo breve oggetto','Gruppo Tecnico']]
    else:
        df_SAP_descrizione = SAP_X_descrizione[['Articolo','Testo breve oggetto','Gruppo Tecnico']]


    df_PLM_descrizione = PLM_BOM[['Articolo','Testo breve oggetto']]  #[['Item Id','Rev Name']]----------------------------------------------------------------------Qui scrive la tabella ∆ quantity

    codici_non_in_PLM = df_confronto_lev2[df_confronto_lev2['Esito check codice']=='Non in PLM']
    AIR_SAP = codici_non_in_PLM.merge(df_SAP_descrizione,how='left',left_on='Articolo', right_on='Articolo')
    AIR_SAP = AIR_SAP[['Articolo','Testo breve oggetto','Gruppo Tecnico','Esito check codice','Azione','Responsabile','Due date','Status']]
    AIR_SAP.rename(columns={'Testo breve oggetto': 'Descrizione'}, inplace=True)

    codici_non_in_SAP = df_confronto_lev2[df_confronto_lev2['Esito check codice']=='Non in SAP']
    AIR_PLM = codici_non_in_SAP.merge(df_PLM_descrizione,how = 'left',left_on='Articolo', right_on='Articolo')
    AIR_PLM = AIR_PLM[['Articolo','Testo breve oggetto','Gruppo Tecnico_PLM','Esito check codice','Azione','Responsabile','Due date','Status']]
    AIR_PLM.rename(columns={'Testo breve oggetto':'Descrizione'},inplace=True)



    df_confronto_lev2_descrizione = pd.concat([AIR_SAP,AIR_PLM],ignore_index=True)
    df_confronto_lev2_descrizione.drop_duplicates(inplace=True)
    df_confronto_lev2_descrizione.reset_index(drop=True, inplace=True)

    df_confronto_lev2_descrizione_print = df_confronto_lev2_descrizione[['Articolo','Descrizione','Gruppo Tecnico','Gruppo Tecnico_PLM','Esito check codice']]


    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader('Action Item livelli 2', divider = 'red')
        st.write(df_confronto_lev2_descrizione_print)
        st.markdown('**KPI 1: percentuale livelli 2 ok: :green[{:0.1f}%]**'.format(100-kpi_liv2)) #0,. per separatore miglialia

    with col2:
        st.subheader('Venn livelli 2', divider = 'red')
        st.pyplot(fig)
        st.write('Livelli 2 in SAP e **non in PLM**: ',lista_Venn[1]) ### modificato
        st.write('Livelli 2 in comune: ',lista_Venn[0]) ### modificato
        st.write('Livelli 2 in PLM e **non in SAP**: ',lista_Venn[2]) ### modificato


    # Download file csv

    def convert_df(df):
        return df.to_csv(index=False,decimal=',').encode('utf-8')   # messo index=False
    csv = convert_df(df_confronto_lev2_descrizione)
    st.download_button(
        label="Download Registro delle azioni livelli 2 in CSV",
        data=csv,
        file_name='Registro_Azioni_Livelli_2.csv',
        mime='text/csv',
    )
    #------------------------------------------------------------------------------------- Visualizzazione dei flag | 2024 02 06 AB 

    st.subheader('Confronto configurazione flag',divider='blue')

    delta = df_confronto_lev2_descrizione
    codice = st.selectbox('selezionare codice', delta.Articolo.unique())
    d33 = SAP_cap
    daa = PLM

    colonne_d33 = d33.columns
    print_33 = pd.DataFrame(columns=colonne_d33)
    inizio = 0
    fine = 0
    for i in range(len(d33)):
        codice_check = d33.Materiale.iloc[i]
        if codice_check == codice:
            livello = d33['Liv. esplosione'].iloc[i]
            livello = int(livello[-1:])
            if livello == 2:
                print_33 = pd.concat([print_33,d33[i:i+1]])
                continue

            for j in range(i):
                riga = i-j
                livello_check = int(d33['Liv. esplosione'].iloc[riga][-1:])
                if livello_check == livello - 1:
                    inizio = riga
                    break
            
            for k in range(len(d33)-i):
                riga = i+k
                livello_check = int(d33['Liv. esplosione'].iloc[riga][-1:])
                if livello_check == livello:
                    fine = riga 
                    break

            print_33 = pd.concat([print_33,d33[inizio:fine+1]])

    colonne_daa = daa.columns
    print_aa = pd.DataFrame(columns=colonne_daa)
    inizio_aa = 0
    fine_aa = 0
    for i in range(len(daa)):
        codice_check = daa['Numero componenti'].iloc[i]
        if codice_check == codice:
            livello = daa['Liv. esplosione'].iloc[i]
            livello = int(livello[-1:])
            if livello == 2:
                print_aa = pd.concat([print_aa,daa[i:i+1]])
                continue

            for j in range(i):
                riga = i-j
                livello_check = int(daa['Liv. esplosione'].iloc[riga][-1:])
                if livello_check == livello - 1:
                    inizio = riga
                    break
            
            for k in range(len(daa)-i):
                riga = i+k
                livello_check = int(daa['Liv. esplosione'].iloc[riga][-1:])
                if livello_check == livello:
                    fine = riga 
                    break

            print_aa = pd.concat([print_aa,daa[inizio:fine+1]])
            
    #----------------------stampa

    print_33 = print_33[['Liv. esplosione','Materiale','Testo breve oggetto','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Prod.',
                        'Ril.Tecn.','Ril.Ric.','Gruppo Tecnico','Descr. Gruppo Tecnico','Inizio validità','Fine validità']]

    colonne_33 = print_33.columns

    print_aa = print_aa[['Liv. esplosione','Numero componenti','Testo breve oggetto','Qtà comp. (UMC)','Merce sfusa','Rilevante produzione',
                        'Ril. progettazione','Cd.parte di ricambio','Gruppo Tecnico','Descr. Gruppo Tecnico','Inizio validità','Fine validità']] 

    colonne_aa = print_aa.columns

    transcodifica = dict(zip(colonne_aa,colonne_33))
    print_aa = print_aa.rename(columns=transcodifica)

    print_aa['ambiente']='TC'
    print_33['ambiente']='SAP'

    colonne_33 = print_33.columns

    completo = pd.DataFrame(columns=colonne_33)
    completo = pd.concat([print_33,print_aa])
    completo = completo[['ambiente','Liv. esplosione','Materiale','Testo breve oggetto','Qtà comp. (UMC)','MerceSfusa (BOM)','Ril.Prod.',
                        'Ril.Tecn.','Ril.Ric.','Gruppo Tecnico','Descr. Gruppo Tecnico','Inizio validità','Fine validità']]

    def highlight_SAP(s):
        return ['background-color:  #09454B']*len(s) if s.ambiente=='SAP' else ['background-color: black']*len(s)

    #st.subheader('Comparazione',divider='blue')
    st.write('Comparazione flag')
    try:
        st.dataframe(completo.style.apply(highlight_SAP, axis=1),width=2500)
    except:
        st.dataframe(completo, width=2500)


    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------fine confronto flag

with tab4: #Delta qty livelli2

    ##### Confronto qty livelli 2
    # Aggiunto algo per qty

    if livello_1 == 'M':
        df_SAP_des_qty = SAP_M_descrizione[['Liv.','Articolo','Qty','Testo breve oggetto','Gruppo Tecnico']]
    elif livello_1 == 'V':
        df_SAP_des_qty = SAP_V_descrizione[['Liv.','Articolo','Qty','Testo breve oggetto','Gruppo Tecnico']]
    else:
        df_SAP_des_qty = SAP_X_descrizione[['Liv.','Articolo','Qty','Testo breve oggetto','Gruppo Tecnico']]


    df_PLM_2 = R_BOM_input[R_BOM_input['Liv.']==2]


    df_PLM_2['Gruppo Tecnico'] = np.where(df_PLM_2['Gruppo Tecnico']!='NO TITOLO', [stringa + '.' for stringa in df_PLM_2['Gruppo Tecnico'].astype(str)], df_PLM_2['Gruppo Tecnico']) # qui il poblema del gt 3
    #correggo il gruppo 3
    df_PLM_2['Gruppo Tecnico'] = df_PLM_2['Gruppo Tecnico'].replace('3..','03.')

    df_SAP_2=df_SAP_des_qty[df_SAP_des_qty['Liv.']==2]

    df_PLM_2['key'] = df_PLM_2['Articolo']+'|'+df_PLM_2['Gruppo Tecnico']
    sum_PLM_2 = df_PLM_2.groupby(by=['key','Liv.','Articolo','Gruppo Tecnico'], as_index=False).sum()
    
    df_SAP_2['key'] = df_SAP_2['Articolo']+'|'+df_SAP_2['Gruppo Tecnico']
    sum_SAP_2 = df_SAP_2.groupby(by=['key','Liv.','Articolo','Testo breve oggetto','Gruppo Tecnico'],as_index=False).sum()

    out = sum_SAP_2.merge(sum_PLM_2, how='outer',left_on='key',right_on='key')
    out = out.rename(columns={'Qty_x':'Qty_SAP','Qty_y':'Qty_PLM'})
    out['delta'] = out.Qty_SAP - out.Qty_PLM
    out = out[out.delta != 0]

    st.dataframe(out[out.delta.astype(str)!='nan'].drop(columns=['key','Liv._x','Liv._y','Articolo_y','Gruppo Tecnico_y']), width = 2500)