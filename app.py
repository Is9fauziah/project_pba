import streamlit as st

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import networkx as nx # untuk membuat model graf dari input array kalimat
from sklearn.feature_extraction.text import CountVectorizer # class dari scikit-learn untuk mengubah teks menjadi vektor numerik
from sklearn.metrics.pairwise import cosine_similarity # class dari scikit-learn untuk menghitung cosine similarity antara dua vektor
import matplotlib.pyplot as plt # untuk membuat plot / gambar graf
import random
import math


def main():
    st.title("Sistem Automatic Text Summarization Menggunakan Algoritma Textrank")
    masuk_text = st.text_input("Maukkan Text")
            # create stemmer dan stopword removal
    factory = StemmerFactory() #inisialisasi objek dari kelas
    stemmer = factory.create_stemmer()
    stop_factory = StopWordRemoverFactory()

    #mendefinisikan variabel tanda baca dan stopword
    arrTdBaca = string.punctuation 
    arrSwindo = stop_factory.get_stop_words()

    #mendefinisikan fungsi bukannum untuk mengecek apakah string adalah angka
    def bukannum(string):
        pattern = r"[^\d]+" # regex untuk mencocokkan karakter bukan angka
        return re.match(pattern, string) is not None

    def remove_double_newline(data):
        cleaned_data = []
        for text in data:
            cleaned_text = text.replace('\n\n', ' ')
            cleaned_data.append(cleaned_text)
        return '\n'.join(cleaned_data)

    def split_text(text):
        # Mengganti karakter \n dengan string kosong dalam teks
        text = text.replace("\n", "")
        # Menggunakan re.split() dengan regular expression untuk memisahkan kalimat
        pre1 = re.split(r"[.!?]\s", text)
        # Membersihkan setiap kalimat dari whitespace yang tidak diperlukan
        pre1 = [s.strip() for s in pre1 if s != ""]
        # Mengembalikan list kalimat
        return pre1

    #fungsi preproses digunakan untuk meminimalkan jumlah kata\kalimat
    def preproses(pre1):
        hasil=[]
        for i in range(0,len(pre1)):
            kalimat = pre1[i]
            pre2 = []
            # proses stemming
            kalimat = stemmer.stem(kalimat)
            # proses tokenisasi 
            tokens = word_tokenize(kalimat)
            for kata in tokens:
            #proses filtering , stopword dan number removal
                if (kata not in arrTdBaca) and (kata not in arrSwindo) and bukannum(kata):
                    pre2.append(kata)
                hasil.append(' '.join(pre2)) 
        
        return hasil

    def create_graph(sentences, preprocessed_sentences): #(bentuk input dari 2 parameter adalah list dari string)
        # Membuat objek graph dari class nx.Graph
        graph = nx.Graph()

        # Menambah setiap kalimat sebagai node baru
        for i, sentence in enumerate(sentences):
            graph.add_node(i, kalimat=sentence)

        # Mengubah setiap kalimat menjadi vektor numerik menggunakan metode BOW dengan bantuan library
        vectorizer = CountVectorizer()
        sentence_vectors = vectorizer.fit_transform(preprocessed_sentences)
        #print(sentence_vectors) 
        
        # Mendapatkan daftar kata-kata dari CountVectorizer
        #kata_kata = vectorizer.get_feature_names_out()
        #print(kata_kata)

        # Menambahkan edge di antara setiap pasang kalimat
        for i, node1 in enumerate(graph.nodes()):
            for j, node2 in enumerate(graph.nodes()):
                if i == j:
                    continue
                
                # Menghitung cosine similarity antara dua kalimat
                similarity = cosine_similarity(sentence_vectors[i], sentence_vectors[j])[0][0]
                #print(similarity)
                #print(f"similarity antara kalimat {i} dan {j}: {similarity}")
                # Menambahkan edge dengan bobot yang setara dengan cosine similarity, jika cosine similarity dari 0
                if similarity > 0:
                    graph.add_edge(i, j, weight=similarity)

        return graph

    toleransi = 1/10000  # batas toleransi untuk error
    debug = {'textrank': False, 'textrank2': False}  # dictionary yang berisi boolean value untuk menyalakan/mematikan debug

    # Fungsi perhitungan textrank dan d sebagai faktor damping yang diset 0.85
    def textrank(graph, d=0.85):
        nsimpul = []  # list yang menyimpan semua simpul dan nilai TextRank dari setiap simpul
        s = [random.randint(1, 3) for x in range(len(graph.nodes))]  # list yang menyimpan nilai TextRank awal dari setiap simpul diset random
        iterasi = 0  # menyimpan jumlah iterasi yang dilakukan saat perhitungan TextRank.
        ilanjut = True  # boolean yang menandakan apakah iterasi dilanjutkan atau tidak

        # """
        # Iterasi utama TextRank dimulai dengan loop while ilanjut. 
        # Di dalam loop ini, setiap simpul dalam graf diperbarui dengan 
        # nilai TextRank yang dihitung.
        # """
        # print(s)
        while ilanjut:
            if debug['textrank2']:
                print('iterasi', iterasi)
            nsimpul = []
            # """
            # for i in graph.nodes() digunakan untuk mengiterasi setiap simpul dalam graf.
            # """
            for i in graph.nodes():
                wij = 0  # bobot antara simpul j dan simpul i
                wjk = 0  # total bobot dari semua simpul yang terhubung dengan simpul j.
                sigma = 0

                # """
                # for j in graph.neighbors(i) digunakan untuk mengiterasi setiap simpul 
                # tetangga dari simpul i. Di dalam loop ini, bobot wij dan wjk diperbarui 
                # dengan bobot yang sesuai dari graf.
                # """
                for j in graph.neighbors(i):
                    if debug['textrank']:
                        print("simpul", i , "dan simpul", j)
                    wij = graph[j][i]['weight'] 
                    if debug['textrank']:
                        print("wij", wij)
                    wjk = sum(graph[i][j]['weight'] for i in graph.neighbors(j))
                    if debug['textrank']:
                        print("wjk", wjk)
                        print("s[", j, "] = ", s[j]) # s[j] adalah nilai TextRank dari simpul j
                    sigma += (wij * s[j]) / wjk  
                    if debug['textrank']:
                        print("sigma", sigma)
                # sigma
                if debug['textrank']:
                    print("wij", wij, "wjk", wjk)
                    print("sigma", sigma)
                if wjk > 0:
                    # """
                    # Variabel txtrank menyimpan 
                    # nilai TextRank yang dihitung berdasarkan faktor damping (d), 
                    # bobot (wij), dan nilai TextRank tetangga (s[j]).
                    # """
                    txtrank = (1 - d) + d * sigma
                    if debug['textrank']:
                        print("s[i] = s[", i, "] = ", s[i])
                        print('txt', txtrank)

                # hitung error
                error = math.fabs(txtrank - s[i])
                if error > toleransi:
                    s[i] = txtrank
                elif i == (len(graph.nodes) - 1):
                    ilanjut = False
                    graph.nodes[i]['nilai'] = txtrank
                nsimpul.append([i, graph.nodes[i]])
                graph.nodes[i]['nilai'] = txtrank

            iterasi += 1
            if iterasi == 100:
                break
        return nsimpul

    # untuk mengurutkan Nilai Textrank tertinggi hingga terendah 
    def descending_sort(node):
        for t in range(0, len(node)):
            temp = t
            for i in range(1 + t, len(node)):
                if node[temp][1]['nilai'] < node[i][1]['nilai']:
                    temp = i
            node[t], node[temp] = node[temp], node[t]
        return node

    # Untuk menampilkan setengah dari dari panjang jumlah kalimat berdasarkan nilai textrank tertinggi
    def get_top_ranked_graphs(graf_list):
        top_ranked_graphs = []
        for graf in graf_list:
            top_ranked_nodes = descending_sort(graf)[:len(graf)//2]
            top_ranked_graphs.append(top_ranked_nodes)
        return top_ranked_graphs

    # Mengubah graf menjadi string
    def get_sentences(graf):
        sentences = []
        for node in graf:
            kalimat = node[1]['kalimat']
            sentences.append(kalimat)
        return '. '.join(sentences)
    button_clicked = st.button("lihat hasil")

    if button_clicked:
        st.write("Text yang dimasukkan : ", masuk_text)
        pre1 = split_text(masuk_text)
        hasil_pre = preproses(pre1)
        hasil_pre

        # Membuat graph
        graph = create_graph(pre1,hasil_pre)

        # Mencetak edge pada graph
        st.write(graph.edges)
        st.write(graph)

        # Menampilkan graph dengan layout circular
        #nx.draw_circular(graph, with_labels=True)
        #plt.show()

        #Memanggil Function Textrank
        result = textrank(graph)

        # Menampilkan hasil TextRank
        hasil_listSimpul = []
        for node in result:
            hasil_listSimpul.append(node)
            st.write("Simpul:", node[0])
            st.write("Nilai TextRank:", node[1]['nilai'])
            st.write()

        #print(hasil_listSimpul)

        # Memanggil function descending_sort
        hasil_sort = descending_sort(hasil_listSimpul)
        st.write("hasil_sort :")
        st.write(hasil_sort)
        list_hasil_sort = [hasil_sort]

        # Memanngil function get_top_ranked_graphs
        hasil_perankingan = get_top_ranked_graphs(list_hasil_sort)
        st.write("hasil_Perankingan :")
        st.write(hasil_perankingan)

        # Memanngil function get_sentences(graf_0)
        graf_0 = hasil_perankingan[0]
        st.write("hasil_teks_rangkuman_sistem :")
        st.write(get_sentences(graf_0))




    # # Mempreproses text input
    # input_text = input("Masukkan teks yang akan diproses: ")
    

if __name__ == "__main__":
    main()
