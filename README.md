 # BERT ALGORİTMASI KULLANILARAK ÇOKLU METİN SINIFLANDIRMA
  Bu projede model eğitiminde kullanılan veri seti 90 bin adet veriden oluşmaktadır. Veriler istekleri ve etiketlerini içermektedir.Bu veriler 6 kategoriden oluşmaktadır. Bu kategoriler "iptal", "hesap" , "kredi kartı", "iade",müşteri hizmetleri" ve "kredi" başlıklarını içerkmektedir.

## BERT ALGORİTMASI NEDİR ?
  BERT (Google Bidirectional Encoder Representations from Transformers) Google'firmasının doğal dil işleme modellerini geliştirmek üzere 2018 yılında piyasaya sürdüğü ve kullandığı sinir ağı algoritmasıdır. Bu algoritmanın diğer algoritmalardan en büyük farkı cümleyi tek bir noktadan değil,hem sağdan hem da soldan işleyerek anlamlı veri çıkarmaya çalışmasıdır. BERT yalnızca cümleyi her iki taraftan taramıyor. Ayrıca veriler ile eğitim sırasında token işlemlerinde tokenleştirilen kelimeleri cümlelerin arasında farkı yerelere koyup tekrar tekrar sinir ağına gönderiyor.Kısaca örneklemek gerekirse,Google arama motoruna Istanbul yazmak yerine "Megakent" veya "Türkiye'nin en gelişmiş şehri" yazmak da bize Google'tarafından Istanbul yanıtının dönmesini hedeflemektedir.

## BERT PREPROCESSİNG SÜRECİ
 Ön işleme sürecinde SimpleTransformers kütüphanesinde faydalanılmıştır. Yukarıda da tanımlanan veri setinin token süreci berttokenizer kütüphanesi kullanılarak ve https://s3.amazonaws.com/models.huggingface.co/bert/dbmdz/bert-base-turkish-128k-uncased/vocab.txt 'dan alınan kelime örneklerine göre tamamlanmıştır. Token işleminin ardından verilerin kategori bölümü dummy kütüphanesi kullanılarak 0 ve 1 cinsinden ifade edilebilecek hale getirilmiştir.Ardından eğitime hazır hale gelen veri seti %80'e %20 train/test olarak ayrılmıştır.Train verisi train batch size: 2, LR ( learning rate )= 0.00001, epoch : 3 değeri boyunca sinir ağından geçirilmiştir. Eğitim sürecinde Google'ın hizmete sunduğu Colab kullanılmıştır. Eğitimde kullanılan ekran kartı TESLA T40 modelidir. Eğitim bu ekran kartı ile ortalama 4 saatte tamamlanmıştır. Eğitim süreci tamamlanan modelin skorları aşağıda gösterilmektedir.

![loss/epoch](https://github.com/aliciplak95/summarify_case/blob/master/Test/lossepoch.png)
![Precision/Recall/F1score](https://github.com/aliciplak95/summarify_case/blob/master/Test/precisionrecallf1score.png) 

## FASTAPI
 Projenin API tarafı FASTAPI tarafından gerçekleşmiştir.JQUERY / HTML  kullanılarak web sayfasından API üzerinden fastapi (python script) dosyasına istek JSON formatında gönderilip verileri python'da işlendikten sonra html'e tekrar aynı formatta alınmıştır. Alınan sınıflama türü ekranda gösterilerek api çalışmasını tamamlamaktadır.


