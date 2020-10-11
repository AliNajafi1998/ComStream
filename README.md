# ComStream ![](https://img.shields.io/badge/Python-3-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=2bbc8a) ![](https://img.shields.io/badge/NLP-TopicDetection-informational?style=flat&logo=data:image/svg%2bxml;base64,<BASE64_DATA>)
<br>

**In this project, we implemented a topic detection system on Twitter. This system reads tweets from a data stream, and assigns them to one of the existing clusters
or a new one. Each cluster acts as an agent, which makes the proposed approach a multi-agent system. There is also a coordinator, who monitors the whole system and 
coordinates the agents. This code has been written in python using several libraries such as ```pandas and colorama``` . The proposed approach has been experimented on two datasets:
The [COVID-19](https://www.kaggle.com/thelonecoder/labelled-1000k-covid19-dataset) and the [FA CUP](http://socialsensor.iti.gr/results/datasets/72-twitter-tdt-dataset). This project has been explained with greater detail in 
a paper, publicly available in ... .**

## System Overview
![logo](https://github.com/AliNajafi1998/ComStream/blob/dev-algo/ComStream.jpg)



## How to use the code ? :hugs:

Data file must be a pandas DataFrame in pickle format having these columns : 

 - text
 - created_at
 - status_id

For example : 
![example](https://github.com/AliNajafi1998/ComStream/blob/dev-algo/data-example.png)

**warning: Data must be sorted based on created_at in ascending order**






### Contributers:

- [Ali Najafi](https://github.com/AliNajafi1998)
- [Araz.G Shilabin](https://github.com/ArazShilabin)
- [Ali Mohammad Pur](https://github.com/alimpfard)
- [Meysam Asgari](https://github.com/MesiA)
- [Rahim Dehkharghani](https://github.com/rdehkharghaniUB)
