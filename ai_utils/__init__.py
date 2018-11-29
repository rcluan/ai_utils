import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

__path__ = __import__('pkgutil').extend_path(__path__, __name__)


def load_dataset(filename):

    file_ext = filename.split('.')[1]

    if(file_ext == 'csv'):
        dataset = pd.read_csv(filename, header=0, index_col=0)
    else:
        dataset = pd.read_excel(filename, header=0, index_col=0)

    return dataset


def preprocess(dataset, reset_index=False, fill_na_columns=[], drop_columns=[], axis=1):
    if(reset_index):
        dataset.reset_index(inplace=True)

    for column in fill_na_columns:
        dataset[column].fillna(0, inplace=True)

    dataset.drop(drop_columns, inplace=True, axis=axis)

# https://www.kaggle.com/stefanie04736/simple-keras-model-with-k-fold-cross-validation
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html


def k_fold(k, x_data, y_data, shuffle=True):
    return StratifiedKFold(n_splits=k, shuffle=shuffle).split(x_data, y_data)


def plot_pred(y_true, y_pred, save_to="~/pred.png", labels=['x', 'y'], line_width=[1, 1]):

    plt.plot(y_true, label='True', linewidth=line_width[0])
    plt.plot(y_pred, label='Prediction', linewidth=line_width[1])

    # Plot labels etc.
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    #plt.xticks([i for i in range(0, len(processes), 250)], processes)
    plt.savefig(save_to)
    plt.show()


"""
    Example:
    send_mail(
        "example@gmail.com",
        "123456",
        "smtp.google.com:587",
        "john@live.com",
        "Report",
        "Sending the report.",
        [str(path + "report.pdf")],
        "doe@gmail.com"
    )
"""

def send_mail(username, password, smtphost, to, subject, msg_body, attachments=None, cc=None):
    try:
        msg = MIMEMultipart()
        message = msg_body

        msg['From'] = username
        msg['To'] = to

        if cc is not None:
            msg['Cc'] = cc

        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        if attachments is not None:
            for f in attachments:
                f = os.path.expanduser(f)
                print(f)
                with open(f, "rb") as fil:
                    part = MIMEApplication(
                        fil.read(),
                        Name=basename(f)
                    )
                # After the file is closed
                part['Content-Disposition'] = 'attachment; filename="%s"' % basename(
                    f)
                msg.attach(part)

        server = smtplib.SMTP(smtphost)
        server.starttls()
        server.login(username, password)
        server.sendmail(msg['From'], msg['To'], msg.as_string())
        server.quit()
        print("Successfully sent email to %s:" % (msg['To']))
    except Exception as e:
        print("Error when sending email %s" % e)
