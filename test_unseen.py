from detector import categorize_url

urls_to_test = [
    "http://36.88.136.194:37295/i",
    "http://110.138.189.142:53741/i",
    "http://120.28.196.250:55255/bin.sh",
    "http://113.120.58.181:40639/i",
    "http://112.198.179.93:52548/bin.sh",
    "http://36.88.136.194:37295/bin.sh",
    "http://105.187.45.37:57187/bin.sh",
    "http://84.214.80.208:47210/i",
    "http://111.88.7.48:38110/i",
    "http://182.117.77.160:50919/bin.sh",
    "http://61.174.126.73:45875/i",
    "http://103.70.87.32:44801/i",
    "http://111.88.7.48:38110/bin.sh",
    "http://123.185.64.59:55925/i",
    "http://123.11.177.223:39159/i",
    "http://103.203.210.102:33865/bin.sh",
    "http://180.190.202.146:56449/bin.sh",
    "http://59.97.250.142:43926/i",
    "http://103.70.87.32:44801/bin.sh",
    "http://119.116.38.254:36146/i",
    "http://61.92.220.176:37567/i",
    "http://177.86.229.12:47177/i",
    "http://120.28.194.92:45133/bin.sh",
    "http://105.186.189.10:43777/i",
    "http://119.116.38.254:36146/bin.sh",
    "https://houseless.com.br/wp-admn/Mesterlrer.qxd",
    "http://61.92.220.176:37567/bin.sh",
    "http://222.127.251.99:49855/bin.sh",
    "http://houseless.com.br/wp-admn/Milieuteknik.fla",
    "http://estudio315.com.br/cbi/Hjertekvababbelse.psd",
    "https://files.catbox.moe/7si03f.js",
    "https://adcentral.com.mx/administrator/new111.png",
    "http://103.249.199.5:51850/bin.sh",
    "http://42.235.138.144:52033/bin.sh",
    "http://216.126.86.14:38209/bin.sh",
    "http://177.86.229.12:47177/bin.sh",
    "http://36.69.80.48:42508/bin.sh",
    "http://27.192.233.39:53271/bin.sh",
    "http://138.204.196.254:53068/i",
    "http://94.156.152.4/yuxwoctober.png",
    "http://94.156.152.4/rumpnovember.png",
    "http://123.185.64.59:55925/bin.sh",
    "http://125.45.56.225:45429/bin.sh",
    "https://github.com/3proxy/3proxy/releases/download/",
    "http://103.191.243.96/ezo",
    "http://217.60.38.14/Work/SOA_.ps1",
    "http://217.60.38.14/Work/PO.ps1",
    "http://217.60.38.14/Work/secured_stub.ps1",
    "http://112.198.238.18:39166/bin.sh",
    "http://103.192.204.30:56111/i",
    "http://125.106.99.57:53331/i",
    "http://192.3.73.173/25/createdbest.vbs",
    "http://201.79.108.167:8081/main_arm7",
    "http://201.79.108.167:8081/main_mips",
    "http://201.79.108.167:8081/main_mpsl",
    "http://112.198.238.18:39166/i",
    "http://42.232.236.6:43947/bin.sh",
    "http://196.190.133.180:38250/i",
    "http://105.224.249.47:51171/bin.sh",
    "http://61.52.99.41:44440/i",
    "http://125.160.141.120:59594/i",
    "http://103.19.49.134:32942/i",
    "http://113.230.192.195:59300/i",
    "http://182.121.114.249:56072/bin.sh",
    "http://120.43.54.135:33228/i",
    "http://200.115.102.246:52425/bin.sh",
    "http://112.93.139.171:38849/bin.sh",
    "http://182.34.62.62:41281/i",
    "http://122.136.26.154:41735/i",
    "http://103.19.49.134:32942/bin.sh",
    "http://crm.lanhsin.eu/si/cmrcdAd.txt",
    "https://kronemedia.de/wp-content/plugins/wpqlxe/",
    "http://124.94.108.115:41401/i",
    "http://105.228.63.209:40587/i",
    "http://45.194.88.27:56446/i",
    "http://196.190.11.194:43393/i",
    "http://45.194.88.27:56446/bin.sh",
    "http://123.14.38.137:60178/bin.sh",
    "https://www.google.com/search?q=python",
    "https://github.com/explore",
    "https://www.linkedin.com/login",
    "https://www.microsoft.com/en-in/",
    "https://www.apple.com/iphone/",
    "https://www.amazon.in/",
    "https://www.wikipedia.org/wiki/Cybersecurity",
    "https://www.youtube.com/",
    "https://www.reddit.com/r/programming/",
    "https://stackoverflow.com/questions",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://www.python.org/downloads/",
    "https://www.mozilla.org/en-US/firefox/",
    "https://www.w3.org/standards/",
    "https://www.ibm.com/products",
    "https://www.oracle.com/database/",
    "https://www.netflix.com/",
    "https://www.canva.com/templates/",
    "https://www.flipkart.com/mobiles",
    "https://www.bbc.com/news"
]

print(f"{'URL':<48} | {'Verdict':<10} | {'Probability':<10}")
print("-" * 75)

safe_count = 0
flagged_count = 0

for url in urls_to_test:
    res = categorize_url(url)
    
    if res['verdict'] == 'safe':
        safe_count += 1
    else:
        flagged_count += 1
        
    print(f"{url:<48} | {res['verdict']:<10} | {res['probability']:.4f}")

print("-" * 75)
print(f"Total: {len(urls_to_test)} | Safe: {safe_count} | Flagged: {flagged_count}")