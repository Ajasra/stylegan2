import runway
from runway.data_types import number, text, image
from example_model import TattoModel

setup_options = {
    'truncation': number(min=0, max=2, step=.01, default=1, description='trunctation for model.'),
    'seed': number(min=0, max=1000000, description='A seed used to initialize the model.')
}
@runway.setup(options=setup_options)
def setup(opts):
    msg = '[SETUP] Ran with options: seed = {}, truncation = {}'
    print(msg.format(opts['seed'], opts['truncation']))
    model = TattoModel(opts)
    return model

@runway.command(name='generate',
                inputs={ 'truncation': number(min=0, max=2, step=.01, default=1, description='trunctation for model.'),
                            'seed': number(min=0, max=1000000, description='A seed used to initialize the model.') },
                outputs={ 'output ': image },
                description='Generates a red square when the input text input is "red".')
def generate(model, args):

    output = model.generate(args['seed'], args['truncation'])
    return {
        'output': output
    }

if __name__ == '__main__':

    runway.run(host='0.0.0.0', port=9000, debug=True)
