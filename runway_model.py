import runway
from runway.data_types import number, text, image, vector, file
from example_model import TattoModel

setup_options = {
    'checkpoint': file(extension='.pkl')
}
@runway.setup(options=setup_options)
def setup(opts):
    model = TattoModel(opts)
    return model

generate_inputs = {
    'z': vector(512, sampling_std=0.5),
    'truncation': number(min=0, max=1, default=0.8, step=0.01)
}

@runway.command(name='generate',
                inputs=generate_inputs, outputs={'image': image},
                description='Generate tattoo based on z-vector')
def generate(model, args):

    output = model.generate(args['z'], args['truncation'])
    return {
        'output': output
    }

if __name__ == '__main__':

    runway.run(host='0.0.0.0', port=9000, debug=True)
