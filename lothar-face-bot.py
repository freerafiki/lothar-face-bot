import logging
import os
import pdb
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from dotenv import load_dotenv
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
load_dotenv()

TOKEN = os.getenv("TOKEN")

logger = logging.getLogger(__name__)

lothars = {
    'ago':['ago', 'genio'],
    'diciommo':['dic', 'diciommo', 'alberto'],
    'facca':['facca', 'fh'],
    'huba':['huba', 'notaio'],
    'lollo':['lollo', 'brorenzo'],
    'moz':['moz'],
    'paggi':['paggi'],
    'palma':['palma'],
    'pecci':['pippo', 'pecci'],
    'scotti':['scotti'],
    'tonin':['ale', 'tonin']
}

lothar_names = ['ago', 'diciommo', 'facca', 'huba', 'lollo', 'moz',
        'paggi', 'palma', 'pecci', 'scotti', 'tonin']

lothars_embeddings = {}

global lothar_art_module

lothar_art_styles = []


def load_hub_module():
    global lothar_art_module
    lothar_art_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


def load_art_styles():
    art_styles_paths = os.listdir('styles')
    lothar_art_styles = []
    for art_style in art_styles_paths:
        if not art_style[0] == ".":
            lothar_art_styles.append(art_style)

def create_folders():
    for lothar_key in lothars:
        folder = "lothar-faces/" + lothar_key
        if not os.path.exists(folder):
            os.mkdir(folder)


def load_embeddings():
    for lothar in lothars:
        lothar_emb = np.loadtxt(f"lothar-embeddings/{lothar}.txt")
        lothars_embeddings[lothar] = lothar_emb


def get_lothar_mentioned(text):
    lothar_found = []
    for lothar_key in lothars:
        if any(lothar in text for lothar in lothars[lothar_key]):
            lothar_found.append(lothar_key)
    return lothar_found


def check_correct_chat(id):
    id = str(id)
    if id == os.getenv("chatlotharid") or id == os.getenv("chatpalmaid"):
        return True
    return False


# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def get_pic_of(update: Update, context: CallbackContext) -> None:
    """Return a photo of someone - needs to be a lothar"""
    if check_correct_chat(update.message.chat.id):
        full_command = ' '.join(context.args)
        lothar_mentioned = get_lothar_mentioned(full_command)
        if len(lothar_mentioned) == 0:
            update.message.reply_text('Non ho trovato nessun lothar - di chi volevi la foto?')
        else:
            for lothar in lothar_mentioned:
                logger.info(f'Chat {update.effective_chat.id} - Photo of {lothar}')
                photo_folder = f"lothar-faces/{lothar}"
                images_path = os.listdir(photo_folder)
                random_index = np.round(np.random.rand() * len(images_path)).astype(int)
                chosen_image = images_path[random_index]
                date_photo = chosen_image[10:12] + "-" + chosen_image[8:10] + "-" + chosen_image[4:8]
                filename = os.path.join(photo_folder, chosen_image)
                update.message.reply_photo(open(filename, 'rb'))
                update.message.reply_text(f'foto di {lothar_mentioned[0]} del {date_photo}')
    else:
        logger.info(f"not allowed in this chat ({update.message.chat.id}), sorry")
    #logger.debug(f"Chat {update.effective_chat.id} - Comando: {command}")


def make_art_from_pic_of(update: Update, context: CallbackContext) -> None:
    """Return a photo of someone - needs to be a lothar"""
    if check_correct_chat(update.message.chat.id):
        full_command = ' '.join(context.args)
        lothar_mentioned = get_lothar_mentioned(full_command)
        if len(lothar_mentioned) == 0:
            update.message.reply_text('Non ho trovato nessun lothar - di chi volevi la foto?')
        else:
            logger.info("making art")
            update.message.reply_text("mi metto all'opera, abbiate pazienza..")
            for lothar in lothar_mentioned:
                logger.info(f'Chat {update.effective_chat.id} - Photo of {lothar}')
                photo_folder = f"lothar-faces/{lothar}"
                images_path = os.listdir(photo_folder)
                random_index = np.round(np.random.rand() * len(images_path)).astype(int)
                chosen_image = images_path[random_index]
                date_photo = chosen_image[10:12] + "-" + chosen_image[8:10] + "-" + chosen_image[4:8]
                filename = os.path.join(photo_folder, chosen_image)
                content_image = plt.imread(filename)
                content_image = cv2.resize(content_image, (480, 640))
                random_index = np.round(np.random.rand() * len(lothar_art_styles)).astype(int)
                # logger.info("random art index: ", random_index, "art styles:", len(art_styles_paths))
                chosen_style = lothar_art_styles[random_index]
                style_image = plt.imread(os.path.join("styles", chosen_style))
                logger.info("chosen style index:", chosen_style)
                logger.info("style image:", style_image[0])
                # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
                content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
                style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
                outputs = lothar_art_module(tf.constant(content_image), tf.constant(style_image))
                stylized_image = np.squeeze(outputs[0].numpy())
                plt.imsave("tmp_art.jpg", stylized_image)
                update.message.reply_photo(open("tmp_art.jpg", 'rb'))
                update.message.reply_text(f'foto di {lothar_mentioned[0]} del {date_photo} rivisitata in stile {chosen_style[:-4]}')
    else:
        logger.info(f"not allowed in this chat ({update.message.chat.id}), sorry")
    #logger.debug(f"Chat {update.effective_chat.id} - Comando: {command}")


def get_mosaic_of(update: Update, context: CallbackContext) -> None:
    """Return the encoding of someone as small image - needs to be a lothar"""
    if check_correct_chat(update.message.chat.id):
        full_command = ' '.join(context.args)
        lothar_mentioned = get_lothar_mentioned(full_command)
        if len(lothar_mentioned) == 0:
            update.message.reply_text('Non ho trovato nessun lothar - di chi volevi la foto?')
        else:
            for lothar in lothar_mentioned:
                logger.info(f'Chat {update.effective_chat.id} - Photo of {lothar}')
                photo_folder = f"lothar-faces/{lothar}"
                width = 480
                height = 640
                big_img = np.zeros((height*3, width*3, 3))
                for i in range(3):
                    for j in range(3):
                        images_path = os.listdir(photo_folder)
                        random_index = np.round(np.random.rand() * len(images_path)).astype(int)
                        chosen_image = images_path[random_index]
                        #date_photo = chosen_image[10:12] + "-" + chosen_image[8:10] + "-" + chosen_image[4:8]
                        filename = os.path.join(photo_folder, chosen_image)
                        small_img = plt.imread(filename)
                        small_img_resized = cv2.resize(small_img, (width, height))
                        big_img[(height*i):(height*(i+1)), (width*j):(width*(j+1)), :] = small_img_resized
                plt.imsave("tmp_mosaic.jpg", big_img / 255)
                update.message.reply_photo(open("tmp_mosaic.jpg", 'rb'))
                #update.message.reply_text(f'mosaico di {lothar_mentioned[0]}')
    else:
        logger.info(f"not allowed in this chat ({update.message.chat.id}), sorry")


def get_embeddings_of(update: Update, context: CallbackContext) -> None:
    """Return the encoding of someone as small image - needs to be a lothar"""
    if check_correct_chat(update.message.chat.id):
        full_command = ' '.join(context.args)
        lothar_mentioned = get_lothar_mentioned(full_command)
        if len(lothar_mentioned) == 0:
            update.message.reply_text('Non ho trovato nessun lothar - di chi volevi la foto?')
        else:
            #np.reshape(known_image_encoding), (16,8)
            for lothar in lothar_mentioned:
                logger.info(f'Chat {update.effective_chat.id} - Embedding of {lothar}:{lothars_embeddings[lothar]}')
                filename = f"lothar-embeddings/{lothar}_emb.jpg"
                update.message.reply_photo(open(filename, 'rb'))
    else:
        logger.info(f"not allowed in this chat ({update.message.chat.id}), sorry")


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def classify_photo(update: Update, context: CallbackContext) -> None:
    if check_correct_chat(update.message.chat.id):
        file = update.message.photo[-1].file_id
        obj = context.bot.get_file(file)
        obj.download("tmp.jpg")
        tmp_image = face_recognition.load_image_file("tmp.jpg")
        recognized_face = face_recognition.face_encodings(tmp_image)
        if recognized_face:
            face_embedding = recognized_face[0]
            results = face_recognition.compare_faces([l_emb for l_emb in lothars_embeddings.values()], face_embedding)
            if np.max(results):
                lothar_found = lothar_names[np.argmax(results)]
                logger.info(results)
                logger.info(lothar_names)
                #logger.info([lll for lll in lothars_embeddings])
                update.message.reply_text(f"trovato {lothar_found} nella foto!")
            else:
                update.message.reply_text("trovato una faccia, ma non lothar")
        else:
            update.message.reply_text("non sembrano esserci facce, o sono un bot ancora troppo stupido")
    else:
        logger.info(f"not allowed in this chat ({update.message.chat.id}), sorry")


def main() -> None:

    logger.info("folders")
    create_folders()
    logger.info("loading the embeddings")
    load_embeddings()
    logger.info("loading the hub module")
    load_hub_module()
    logger.info("loading the art styles")
    load_art_styles()
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(TOKEN)
    #-1001256630978
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("pic", get_pic_of))
    dispatcher.add_handler(CommandHandler("vector", get_embeddings_of))
    dispatcher.add_handler(CommandHandler("mosaic", get_mosaic_of))
    dispatcher.add_handler(CommandHandler("art", make_art_from_pic_of))

    # on non command i.e message - echo the message on Telegram
    #dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))
    dispatcher.add_handler(MessageHandler(Filters.photo, classify_photo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
