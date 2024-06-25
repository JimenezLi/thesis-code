import urllib.request
import calendar
import time


def wsprlive_get(query):
    # put together the request url
    # url = "https://db1.wspr.live/?query=" + urllib.parse.quote_plus(query + " FORMAT JSON")
    url = "https://db1.wspr.live/?query=" + urllib.parse.quote_plus(query)

    # download contents from wspr.live
    contents = urllib.request.urlopen(url).read()

    # return the json decoded data
    return contents.decode("UTF-8")


def wsprlive_my_query(year, month):
    start_month = '{}-{:02d}'.format(year, month)
    end_day = '{}-{:02d}-{:02d}'.format(year, month + 1, calendar.monthrange(year, month + 1)[1])

    # print(start_day, end_day)
    # return

    query = f'''
select
    toStartOfHour(time) as time, toUnixTimestamp(toStartOfHour(time)) as timestamp, max(frequency) as mf
from wspr.rx where
    time between '{start_month}-01 00:00:00' and '{end_day} 23:59:59'
    and ((rx_loc between 'JO31' and 'JO34' and tx_loc between 'EM68' and 'EM71') or (tx_loc between 'JO31' and 'JO34' and rx_loc between 'EM68' and 'EM71'))
group by time
order by time
limit 2000'''

    with open(f'../../res/2month/{start_month}.tsv', mode='w') as f:
        print('time\ttimestamp\tmf', file=f)
        print(wsprlive_get(query), file=f)
        f.close()


if __name__ == "__main__":
    for i in range(2008, 2014):
        for j in range(1, 12, 2):
            try:
                wsprlive_my_query(i, j)
                print(f'Successfully get {i}-{j}')
            except:
                print(f'Fail to get {i}-{j}')
            finally:
                time.sleep(5)
