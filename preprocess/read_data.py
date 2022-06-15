### Get some statistics of the data

import os
app_version_dict = {"ebay": "2.6.0", "viber": "4.3.1", "barebone": "3.", "hmbtned": "4.", "timberiffic": "1.10"}

def read_data(app, fn):
    fp = open(fn, "rb")
    reviews = fp.readlines()
    fp.close()
    count = 0

    for idx, review in enumerate(reviews):
        terms = review.split(str.encode("\t"))
        # print(terms)
        version = terms[2]

        if version.startswith(str.encode(app_version_dict[app])):
            count += 1
    print("Review no. for ", app, " is ", count)

if __name__ == "__main__":
    read_data("hmbtned", "../../selected_data/reviews/air.hmbtned/air.hmbtned_2.txt")

    # com.ebay.mobile/com.ebay.mobile_2.txt
    # com.viber.voip/com.viber.voip_2.txt
    # air.hmbtned/air.hmbtned_2.txt
    # com.alfray.timeriffic/com.alfray.timeriffic_2.txt
    # acr.browser.barebones/acr.browser.barebones_2.txt
