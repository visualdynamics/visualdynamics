# Rescaling a record

A channel recorded with the wrong sensitivity is off by a constant
factor. The acquisition divides the measured volts by the sensitivity
it was given, so data taken with a sensitivity of 100 mV/g from a
sensor whose sensitivity is 10 mV/g reads a tenth of the true level.
The correction is to multiply the channel by the recorded sensitivity
over the correct one, 100 / 10 = 10 here.

## Rescale on the data bar

Select a time history, or pick some of its channels in its grid, and
choose **Rescale** on the data bar. The pane beside the plot has the
factor's settings:

- **Given as**: *Factor*, to type the factor itself, or
  *Sensitivities*, to give the two sensitivities and let the factor be
  worked out from them. Only the fields the choice reads are shown.
- **Factor**: what the channels are multiplied by. A negative factor
  also flips the sign, which corrects a sensor mounted backwards.
- **Recorded with** and **Correct**: the sensitivity the data was
  recorded with and the sensor's correct one, in mV per engineering
  unit.

Under them are the factor that applies, how many of the record's
channels it applies to, and the largest value of those channels before
and after. The plot draws the channels as recorded and as rescaled on
the same axes, so the change can be checked before it is made.

**Apply Rescale** makes a new time history named for the source
followed by *Rescaled* (`Run 3 Rescaled`), linked beside it. The
channels not picked are copied unchanged. The new record keeps the
source's clock, its channels, the averaging and shock windows, the
filter and truncation settings and the channel roles, so spectra, PSDs
and FRFs computed from it are set up the same way as from the source.
The source is left as it was; delete it once the rescaled record has
replaced it.

The channel table is not changed. If its sensitivity is wrong too,
correct that cell in the table separately. Editing a sensitivity in
the table never rescales data, since the edit can just as well be
recording a sensor's new calibration for the next test.

## From a script

    import visualdynamics

    project = visualdynamics.Project.open('test.vdyn')
    project.rescale('Run 3', recorded_sensitivity=100.0,
                    correct_sensitivity=10.0, records=[4, 5])
    project.rescale('Run 3', -1.0, records=[2])

`project.rescale` takes a factor, or the two sensitivities in its
place, and `records=` limits it to some channels, counting from zero.
`history.rescale(factor, records)` returns the rescaled record without
adding it to the project, and `core.rescale.sensitivity_factor` gives
the factor two sensitivities make.
