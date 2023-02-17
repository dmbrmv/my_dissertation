def triple_position_number(value: int):

    if len(str(value)) == 1:
        return f'00{value}'
    elif len(str(value)) == 2:
        return f'0{value}'
    elif len(str(value)) == 3:
        return f'{value}'
    else:
        raise Exception(f'Something wrong with number {value} !')
