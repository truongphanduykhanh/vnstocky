import datetime
from dateutil.relativedelta import relativedelta


def get_hours_minutes_seconds(timedelta):
    '''
    Convert time delta to hours, minutes, seconds

    Parameters
    ----------
    timedelta : datetime.timedelta
        time delta between two time points. Ex: datetime.timedelta(0, 9, 494935)

    Returns
    -------
    three integer objects corresponding to number of hours, minutes and seconds
    '''
    total_seconds = timedelta.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds - (hours * 3600)) // 60
    seconds = total_seconds - (hours * 3600) - (minutes * 60)
    return hours, minutes, seconds


def get_last_dates(input_time,
                   input_format,
                   output_format='%Y%m%d',
                   periods=1,
                   mode='month'):
    '''
    Get list of last dates of months or weeks

    Parameters
    ----------
    input_time : str
        input time. Ex: '202106' or '20210630'
    input_format : str
        format of input time. Ex: '%Y%m' or '%Y%m%d'
    output_format : str
        format of output time. Ex: '%Y%m' or '%Y%m%d'
    periods : int
        number of time periods wanted to look back. Ex: 2
    mode : str
        mode to look back. Ex: 'month' or 'week'

    Returns
    -------
    list of str
        list of last dates of months or weeks. Ex: ['20210630', '20210531']
    '''
    if mode not in ['month', 'week']:
        raise ValueError('mode should be either "month" or "week".')

    input_time = datetime.datetime.strptime(input_time, input_format)
    dates = []
    if mode == 'month':
        for i in range(periods):
            date = input_time - relativedelta(months=i)
            date = date.replace(day=28) + relativedelta(days=4)
            date = date - relativedelta(days=date.day)
            date = date.strftime(output_format)
            dates.append(date)
    else:
        for i in range(periods):
            date = input_time.replace(day=28) + relativedelta(days=4)
            date = date - relativedelta(days=date.day)
            date = date - relativedelta(weeks=i)
            date = date.strftime(output_format)
            dates.append(date)
    return dates


def get_last_dates_range(start_date,
                         end_date,
                         input_format='%Y%m%d',
                         output_format='%Y%m%d'):
    '''
    Get list of last dates from starting date to ending date included.

    Parameters
    ----------
    start_date : str
        Starting date. Ex: '20200630'.
        Starting date is not neccessarily at the end of the month.
        I.e. '20200630' or 20200615 or '20200601' would result the same.
    end_date : str
        Ending date. Ex: '20210630'.
        Ending date is not neccessarily at the end of the month.
        I.e. '20210630' or 20210615 or '20210601' would result the same.
    input_format : str
        Format of input dates. Ex: '%Y%m%d'
    output_format : str
        Format of output dates. Ex: '%Y%m%d'

    Returns
    -------
    list of str
         List of last dates. Ex: ['20210630', '20210531', ...]
    '''
    start_date = get_last_dates(start_date, input_format=input_format)[0]
    end_date = get_last_dates(end_date, input_format=input_format)[0]

    date = start_date
    dates = []
    dates.append(date)
    while date < end_date:
        date = datetime.datetime.strptime(date, output_format)
        date = date + relativedelta(months=1)
        date = date.replace(day=28) + relativedelta(days=4)
        date = date - relativedelta(days=date.day)
        date = date.strftime(output_format)
        dates.append(date)
    return dates
