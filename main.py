# main.py
import logging
from ner_model.model.ckip_pipeline import CkipTransformerPipeline

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    
    sample_texts = [
        "#謝謝🙏 文茜的世界周報 Sisy's World News 陶晶瑩 賈永婕的跑跳人生 胡小禎 李佩甄 隋棠 Sonia Sui 藍心湄 Hsin-Mei Lan 德州媽媽沒有崩潰 瑪麗的象牙塔 MARY in the TOWER 于美人 潘若迪_Funky Dance 李李仁 June Yu 于長君 林柏宏 ØZI Patrick 派翠克 焦凡凡fanfan 婁峻碩 SHOU Melody時尚媽咪",
        "Toyz原先預定26日晚間在浪live開直播，不過後來轉到YouTube開直播。",
        "🌟 合作信箱：toyzpr@gmail.com 🌟 我的生活頻道 ：https://www.youtube.com/c/Toyz69 🌟 我的Instagram：https://www.instagram.com/toyzlol 🌟 我的Facebook：https://www.facebook.com/Toyzlau 🌟 浪live直播：https://www.lang.live/main",
        "☝🏻浪live固定開播日期：2024/4/15(一)起 每周一 21:00 🔎超派直播間浪ID： 1111 還沒有浪live帳號嗎？",
        "活動當日，中獎者本人抵達活動現場需打開浪live投票紀錄以茲證明，未符合當場失去資格，開放給現場候補粉絲入場。",
        "大家好，我爆料浪live女主播id是5058976咪子主播散播色情，我會附上截圖照片證據。",
        "In 2024, everyone is really kind.",
        "柯文哲參選總統時自信滿滿, 如今柯文哲卻被抓進土城看守所。",
        "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
        "China has dismissed the outcome of Taiwan’s elections, saying the DPP does not represent the mainstream public opinion.",
        "我很喜歡Adidas今年出的運動裝，每年一定會購買他們的新款。",
        "My name is Patty Chang.\n我想買許多Nvidia 4070顯卡。",
        "la mer是春香想要很久的保養品 每次都會跟我唸說想要經典乳霜！",
        "登錄發票再抽環保好禮🎁 響應世界地球日，買蒲公英商品滿額登錄發票，即可抽Gogoro、AppleWatch等好禮！",
        "我要買0080這支股票"
    ]
    
    pipeline = CkipTransformerPipeline(
        device_ch=1, 
        device_eng_ner=1, 
        device_eng_pos=1, 
        device_translate=1, 
        batch_size_ch=8, 
        batch_size_en=8,
        model_name='bert_base', 
        eng_ner_model='eng_ontonotes_large', 
        eng_pos_model='eng_vblagoje_pos',
        translate_model='translate'
    )
    
    entities = pipeline.get_named_entities(
        texts=sample_texts,
        use_batch=True,
        max_length=120,
        use_delimiter=False,
        show_progress=False
    )
    
    print("Named Entities:")
    for idx, entity in entities.items():
        print(f"Text {idx}: {entity}")

if __name__ == "__main__":
    main()
