import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import lumicks.pylake as lk
import ipywidgets as widgets
from IPython.core.display import display

def import_File(input_path):
    '''Imports a Bluelake HDF5 file. input_path is the file's path and must be a string.'''
    assert os.path.exists(input_path), 'Sorry, the selected file does not exist.'
    file = lk.File(input_path)
    name = os.path.basename(input_path).split('.')[0]
    folder = os.path.dirname(input_path)
    return folder, file, name

def misspelled_channel(kymo_or_scan,channel):
    assert hasattr(kymo_or_scan, f'{channel}_image'), f'Unknown argument channel = {channel}. Is it misspelled?'

def relate_ChannelColor(channel):
    if channel == 'red': color = 'red'
    if channel == 'green': color = 'lime'
    if channel == 'blue': color = 'blue'
    return color

def make_colormap(seq):
    ''' Returns a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples.
    The floats should be in increasing order and within the interval (0,1).'''
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def extract_Kymo(file):
    '''Extracts the info of interest from a kymo dictionary from a Bluelake HDF5 file.'''
    kymo = list(file.kymos)
    kymo = file.kymos[kymo[0]]
    kymo_t0 = kymo.timestamps[0,0]
    kymo_time = (kymo.timestamps[0,:]-kymo_t0)/1e9
    duration = kymo_time[-1]
    length = kymo.json['scan volume']['scan axes'][0]['scan width (um)']
    time_lines = len(kymo_time)
    max_pixels = kymo.json['scan volume']['scan axes'][0]['num of pixels']
    return kymo, kymo_time, duration, length, time_lines, kymo_t0, max_pixels

def extract_Scan(file):
    '''Extracts the info of interest from a scan dictionary from a Bluelake HDF5 file.'''
    scan = list(file.scans)
    scan = file.scans[scan[0]]
    x_max = round(scan.json['scan volume']['scan axes'][0]['scan width (um)'], 5)
    y_max = round(scan.json['scan volume']['scan axes'][1]['scan width (um)'], 5)
    pixelsizex = scan.json['scan volume']['scan axes'][0]['pixel size (nm)']
    pixelsizey = scan.json['scan volume']['scan axes'][1]['pixel size (nm)']
    if pixelsizex != pixelsizey: print('Warning: Pixels are not squared. {pixelsizex}x{pixelsizey} nm')
    scan_t0 = scan.timestamps[0,0,0]
    frame_rate = round(((scan.timestamps[0,-1,-1] - scan_t0 )/1e9)**(-1), 5)
    return scan, x_max, y_max, frame_rate, scan_t0

def extract_Force(file, downsampled_rate, timeref0):
    '''Extracts the force of the second optical trap from a Bluelake HDF5 file. Downsamples the force dataset to a choosen rate.'''
    force = file.force2x
    force_data = force.data
    force_time = (force.timestamps - timeref0)/1e9
    sample_rate = force.sample_rate
    force_dataset = [force_time, force_data]

    downforce = force.downsampled_by(int(sample_rate/downsampled_rate))
    downforce_data = downforce.data
    downforce_time = (downforce.timestamps - timeref0)/1e9
    downforce_dataset = [downforce_time, downforce_data]

    return force_dataset, downforce_dataset, sample_rate

def export_ScanForce(input_path, downsampled_rate=100):
    '''Exports the force dataset of a scan to a CSV file.'''
    folder, file, name = import_File(input_path)
    scan, x_max, y_max, frame_rate, scan_t0 = extract_Scan(file)
    force_dataset, downforce_dataset, sample_rate = extract_Force(file, downsampled_rate, scan_t0)

    df = pd.DataFrame({'Time (s)' : force_dataset[0], 'Force (pN)' : force_dataset[1]})
    df_down = pd.DataFrame({'downsampled Time (s)' : downforce_dataset[0], 'downsampled Force (pN)' : downforce_dataset[1]})

    outpath = os.path.join(folder,f'{name}_force_{sample_rate}Hz.csv')
    df.to_csv(outpath, index=False)
    outpath = os.path.join(folder,f'{name}_downsampledforce_{downsampled_rate}Hz.csv')
    df_down.to_csv(outpath, index=False)

def export_PhotonsKymo(input_path, channel = 'all'):
    '''Exports a matrix of the summed photon counts for each pixel of a Kymograph.
    The output file is a CSV file containing a 2D-matrix of photon counts.
    The indexes of the matrix correspond:
        First index: Coordinate of the pixel within a scan line of the confocal.
        Second index: Time coordinate.

    Function arguments: input_path, channel
      - input_path is the path to the HDF5 data file. Must be a string.
      - channel refers to the RGB channel to be exported. Must be a string.
            Accepts: 'red', 'green' or 'blue' for a specific matrix
                     'all' to export the matrixes for the splitted channels
                     If no argument is given the function takes 'all' by default.

    NOTE: If the file contains more than one kymograph only the first one is taken into consideration.'''
    folder, file, name = import_File(input_path)
    kymo, kymo_time, duration, length, time_lines, kymo_t0, max_pixels = extract_Kymo(file)

    def saveTXT(kymo, channel, folder, name, duration, length):
        kymo_channel = getattr(kymo, f'{channel}_image')
        outpath = os.path.join(folder,f'{name}_{channel}.csv')
        x_max = duration; y_max = length; z_max = int(kymo_channel.max());
        np.savetxt(outpath, kymo_channel,  fmt='%d', delimiter='\t', newline='\n', comments='',
                   header = f'Kymograph {channel}-channel ASCII Matrix file \nX Amplitude: {x_max} s \nY Amplitude: {y_max} µm \nZ Amplitude: {z_max} a.u.')

    if channel == 'all':
        channels = ['red','green','blue']
        for channel in channels:
            saveTXT(kymo, channel, folder, name, duration, length)
    else:
        misspelled_channel(kymo,channel)
        saveTXT(kymo, channel, folder, name, duration, length)

def export_PhotonsScan(input_path, channel = 'all', frame = 0):
    '''Exports a matrix of the summed photon counts for each pixel of a Scan for a particular frame.
    The output file is a CSV file containing a 2D-matrix of photon counts.
    The indexes of the matrix correspond:
        First index: Coordinate of the pixel within a scan line of the confocal.
        Second index: Time coordinate.

    Function arguments: input_path, channel
      - input_path is the path to the HDF5 data file. Must be a string.
      - channel refers to the RGB channel to be exported. Must be a string.
            Accepts: 'red', 'green' or 'blue' for a specific matrix
                     'all' to export the matrixes for the splitted channels
                     If no argument is given the function takes 'all' by default.

    NOTE: If the file contains more than one scan only the first one is taken into consideration.'''
    folder, file, name = import_File(input_path)
    scan, x_max, y_max, frame_rate, scan_t0 = extract_Scan(file)

    def csvheader(channel,frame,x_max,y_max,z_max):
        main = f'Scan {channel}-channel (frame {frame+1}) ASCII Matrix file'
        specific = f'\nFrame Rate: {frame_rate} fps \nX Amplitude: {x_max} µm \nY Amplitude: {y_max} µm \nZ Amplitude: {z_max} a.u.'
        header = main + specific
        return header

    def savePhotons(scan,channel,frame,name,folder,x_max,y_max):
        scan_channel = getattr(scan, f'{channel}_image')
        outpath = os.path.join(folder,f'{name}_{channel}_frame{frame+1}.csv')
        z_max = int(scan_channel.max());
        np.savetxt(outpath, scan_channel[frame,:,:], fmt='%d', delimiter='\t', newline='\n', comments='',
                                                    header = csvheader(channel,frame,x_max,y_max,z_max))
    if channel == 'all':
        channels = ['red','green','blue']
        if frame == 'all':
            for channel in channels:
                for i in range(0,scan.num_frames):
                    frame = i
                    savePhotons(scan,channel,frame,name,folder,x_max,y_max)
        else:
            frame = frame-1 #because frames start at 1 but indexes start as 0
            assert 0 < frame <= scan.num_frames, f'Sorry, frame range goes from 1 to {scan.num_frames}.'
            for channel in channels:
                savePhotons(scan,channel,frame,name,folder,x_max,y_max)
    else:
        misspelled_channel(scan, channel)
        if frame == 'all':
            for i in range(0,scan.num_frames):
                frame = i
                savePhotons(scan,channel,frame,name,folder,x_max,y_max)
        else:
            frame = frame-1 #because frames start at 1 but indexes start as 0
            assert 0 < frame <= scan.num_frames, f'Sorry, frame range goes from 1 to {scan.num_frames}.'
            savePhotons(scan,channel,frame,name,folder,x_max,y_max)

def align_axis(ax, align_to_ax, axis):
    '''Aligns the choosen axis of an ax to the axis of the align_to_ax in the figure'''
    posn_old, posn_target = ax.get_position(), align_to_ax.get_position()
    if axis == 'x' : ax.set_position([posn_target.x0, posn_old.y0, posn_target.width, posn_old.height])
    if axis == 'y' : ax.set_position([posn_old.x0, posn_target.y0, posn_old.width, posn_target.height])

def add_button_savePNG(fig, folder, outname, dpi, transparent = False):
    '''Button to save the image to PNG '''
    outpath = os.path.join(folder,f'{outname}.png')
    button = widgets.Button(description = 'Save PNG')
    output = widgets.Output()
    display(button, output)
    def on_button_clicked(b):
        fig.savefig(outpath, dpi=dpi, transparent=transparent)
        with output: print('Done.')
    button.on_click(on_button_clicked)

def add_button_savePDF(fig, folder, outname):
    '''Button to save the image to PDF '''
    outpath = os.path.join(folder,f'{outname}.pdf')
    button = widgets.Button(description = 'Save PDF')
    output = widgets.Output()
    display(button, output)
    def on_button_clicked(b):
        fig.savefig(outpath)
        with output: print('Done.')
    button.on_click(on_button_clicked)

def add_button_saveTIFF(kymo, folder, outname):
    '''Button to save the image as TIFF '''
    outpath = os.path.join(folder,f'{outname}.tiff')
    button = widgets.Button(description = 'Save TIFF')
    output = widgets.Output()
    display(button, output)
    def on_button_clicked(b):
        kymo.save_tiff(outpath)
        with output: print('Done.')
    button.on_click(on_button_clicked)

def add_button_saveCSV(kymo_data, channel, folder, outname):
    '''Button to save data to CSV '''
    button = widgets.Button(description = f'Save CSV {channel}')
    output = widgets.Output()
    display(button, output)
    def on_button_clicked(b):
        outpath = os.path.join(folder,f'{outname}.csv')
        np.savetxt(outpath, kymo_data,  fmt='%d', delimiter='\t', newline='\n', comments='',
                   header = f'Kymograph {outname}')
        with output: print('Done.')
    button.on_click(on_button_clicked)

def plot_Kymograph(input_path, channel = 'green', color = True, figsize = None, t_start = None, t_end = None, colorbar = True, orientation = 'vertical', threshold = None, force_correlation = True, downsampled_rate = 100, legend = False, axis = False, label_fontsize = 12, L=0.08, R=0.96, T=0.93, B=0.08, Vspace=0.03, title = True, transparentBack = True, dpi = 300):
    '''
    Returns an image of the kymograph for a particular channel.
    Function arguments: input_path, channel
        - input_path is the path to the HDF5 data file. Must be a string.
        - channel refers to the RGB channel to be represented. Must be a string.
                Accepts: 'red', 'green' or 'blue' for a specific matrix
                         If no argument is given the function takes 'green' by default.
        - color refers to the colormap of the image.
                If False the colormap will be greyscale.
                If True it will have a color related to the channel.
                If no argument is given the function takes True by default.
        - colorbar allows the user to choose whether to add the colorbar to the image or not.
                If False no colorbar is added.
                If True a color bar will be added to the image.
                If no argument is given the function takes True by default.
        - savecolorbar allows the user to choose whether to save the colorbar in a separate image.
                If False the colorbar will not be saved seperatly.
                If colorbar = False, the colorbar will not be saved nor represented.
                If no argument is given the function takes False by default.
        - threshold allows the user to define the upper limit of the colormap to a particular value.
                If None, the upper limit is taken as the maximum value of photon counts.
                If no argument is given the function takes None by default.
        - axis allows the user to choose whether to add axis to the image or not.
                If False axis are removed from the image.
                If True axis are kept in the image.
                If no argument is given the function takes True by default.
        - label_fontsize allows the user to define a fontsize for the image axis labels.
                If no value is given, the function takes size 12 by default.
        - L = 0.096 , R = 0.942 , T = 0.949 , B = 0.126 are the Left, Right, Top, Bottom spacings on the final image.
                These values can be changed by calling the arguments and defining a new value.

    NOTE: If the file contains more than one kymograph only the first one is taken into consideration.    '''
    folder, file, name = import_File(input_path)
    kymo, kymo_time, duration, length, time_lines, kymo_t0, max_pixels = extract_Kymo(file)
    misspelled_channel(kymo,channel)
    kymo_channel = getattr(kymo, f'{channel}_image')

    if threshold == None: threshold = kymo_channel.max()

    if color == False: colorrange = plt.cm.gray
    else:
        c_color = relate_ChannelColor(channel)
        c = mcolors.ColorConverter().to_rgb
        colorrange = make_colormap([c('black'), c(f'{c_color}')])

    frame_rate = duration/time_lines
    if t_start != None:
        filtr = int(t_start/frame_rate) # get starting time line
        kymo_channel = kymo_channel[:,filtr:]
        x0 = t_start
    else: x0 = 0
    if t_end != None:
        filtr = int(t_end/frame_rate)
        kymo_channel = kymo_channel[:,:filtr]
        xf = t_end
    else: xf = duration

    rows = 2 if force_correlation == True else 1
    heights = [5,1] if force_correlation == True else [1]

    fig, axs = plt.subplots(nrows = rows, ncols = 1, figsize = figsize, gridspec_kw = {'width_ratios':[1], 'height_ratios':heights})
    if rows == 1: axs = [axs]
    if title == True: fig.suptitle(f'{name}_{channel}')
    plt.subplots_adjust(left=L, bottom=B, right=R, top=T, hspace=Vspace)
    im = axs[0].imshow(kymo_channel, extent = (x0, xf, length, 0), aspect = 'auto', vmin = 0, vmax = threshold, cmap = colorrange)

    if axis == True:
        axs[0].set_ylabel('Position (µm)', fontsize = label_fontsize)
        axs[0].tick_params(direction = 'in', top='on', right='on')
        if force_correlation == False: axs[0].set_xlabel('Time (s)', fontsize = label_fontsize)
        else: axs[0].set_xticks([])
    else: axs[0].axis('off')

    if force_correlation == True:
        force_dataset, downforce_dataset, sample_rate = extract_Force(file, downsampled_rate, kymo_t0)
        axs[1].scatter(force_dataset[0], force_dataset[1], s=0.1, c='darkgrey', label = f'{sample_rate} Hz')
        axs[1].scatter(downforce_dataset[0], downforce_dataset[1], s=0.2, c='dimgrey', label = f'{downsampled_rate} Hz')
        axs[1].set_ylabel('F (pN)', fontsize = label_fontsize)
        axs[1].set_xlabel('Time (s)', fontsize = label_fontsize)
        axs[1].tick_params(direction = 'in', top='on', right='on')
        axs[1].set(xlim=(x0, xf), ylim=(None, None))
        align_axis(ax = axs[1], align_to_ax = axs[0], axis = 'x')
        if legend == True: axs[1].legend(fontsize=label_fontsize)

    if colorbar == True:
        cbar = plt.colorbar(im, ax = axs[0], orientation = orientation, pad = 0.01, fraction = 0.05)
        cbar.ax.tick_params(direction='in')
        if force_correlation == True: align_axis(ax = axs[1], align_to_ax = axs[0], axis = 'x')

    add_button_savePNG(fig, folder, f'{name}_{channel}', dpi, transparent = transparentBack)
    add_button_saveTIFF(kymo, folder, f'{name}_RGB')

    print(f'Kymograph contains {len(kymo_channel[0])} scan lines')
    print(f'Scanning rate: {(1/frame_rate)} lines/second')
    if force_correlation == True:
        print(f'Force Sample Rate: {sample_rate} Hz')
        print(f'Force Downsampled Rate: {downsampled_rate} Hz')

    return kymo

def plot_PhotonProfile(input_path, channels = ['red', 'green', 'blue'], figsize=(9,5),  threshold = None, show_CSVbuttons = True):
    '''
    Plots the photon count profile for a particular time line.
    '''
    folder, file, name = import_File(input_path)
    kymo, kymo_time, duration, length, time_lines, kymo_t0, max_pixels = extract_Kymo(file)

    kymo_channels = {'red': kymo.red_image, 'green':kymo.green_image, 'blue':kymo.blue_image}

    zmax = max(kymo_channels['red'].max(), kymo_channels['green'].max(), kymo_channels['blue'].max())
    if threshold == None: threshold = zmax

    fig,ax = plt.subplots(figsize=figsize)
    def update_plot(tline):
        ax.cla()
        ax.set(xlabel='Pixel', ylabel='Photon Count (a.u.)', ylim = (0, threshold+1), xlim = (0, max_pixels-1))
        ax.tick_params(direction = 'in', top='on', right='on')
        for channel in channels:
            ax.plot(kymo_channels[channel][:,tline], f'{channel}')
        text.value = f'{kymo_time[tline]} s'
        fig.suptitle(f'{name}_{channels}_time=line {tline}/{time_lines-1}')
        button = add_button_savePNG(fig, folder, f'{name}_{channels}_photons-profile@{text.value}-line{box.value}', 350)
        if show_CSVbuttons==True:
            for channel in channels:
                add_button_saveCSV(kymo_channels[channel][:,tline], channel, folder,f'{name}_{channel}_photons-profile@{text.value}-line{box.value}')

    text = widgets.Text(value='My Text', description='Time (s):', disabled=True)
    box = widgets.BoundedIntText(value=0, min=0, max=time_lines-1, step=1, description='Time-line:')

    display(text)
    widgets.interact(update_plot, tline = box, text = text)

def plot_Scan(input_path, figsize = (9,5), channel = 'green', threshold = None, axis = True, label_fontsize = 12, force_correlation = True, downsampled_rate = 100, force_pointsize = 0.1, downs_force_pointsize = 0.1, force_color = 'lightgrey', downs_force_color = 'dimgrey', shade_color = 'yellow', shade_alpha = 0.2, plots_ratios = [2,1], L = 0.096 , R = 0.942 , T = 0.93 , B = 0.126, Vspace=0.01, title = True, showSaveButtons = True, transparentBack = False, dpi = 350):
    ''' Returns the image of a frame of the scan for a particular channel.'''
    folder, file, name = import_File(input_path)
    scan, x_max, y_max, frame_rate, scan_t0 = extract_Scan(file)
    misspelled_channel(scan,channel)
    scan_channel = getattr(scan, f'{channel}_image')

    if threshold == None: threshold = kymo_channel.max()

    c_color = relate_ChannelColor(channel)
    c = mcolors.ColorConverter().to_rgb
    colorrange = make_colormap([c('black'), c(f'{c_color}')])

    rows = 2 if force_correlation == True else 1
    plots_ratios = plots_ratios if force_correlation == True else [1]

    fig, axs = plt.subplots(nrows = rows, ncols = 1, figsize = figsize, gridspec_kw = {'width_ratios':[1], 'height_ratios':plots_ratios})
    if rows == 1: axs = [axs]
    plt.subplots_adjust(left=L, bottom=B, right=R, top=T, hspace=Vspace)
    if axis == True:
        axs[0].set_xlabel('Position (µm)', fontsize = label_fontsize)
        axs[0].set_ylabel('Position (µm)', fontsize = label_fontsize)
        axs[0].tick_params(direction='in')
    else:
        axs[0].axis('off')

    if force_correlation == True:
        force_dataset, downforce_dataset, sample_rate = extract_Force(file, downsampled_rate, scan_t0)
        axs[1].scatter(force_dataset[0], force_dataset[1], s = force_pointsize, c = force_color, zorder=1)
        axs[1].scatter(downforce_dataset[0], downforce_dataset[1], s = downs_force_pointsize, c = downs_force_color, zorder=2)
        axs[1].set_ylabel('F (pN)', fontsize = label_fontsize)
        axs[1].set_xlabel('Time (s)', fontsize = label_fontsize)
        axs[1].set(xlim=(0,force_dataset[0].max()), ylim=(None,None))
        axs[1].tick_params(direction = 'in', top = True, right = True)

    def update_frame(frame):
        if title == True: fig.suptitle(f'{name}_{channel}_frame{frame}')
        im = axs[0].imshow(scan_channel[frame-1],
                            extent = (0, x_max, y_max, 0), aspect = 'equal',
                            vmin = 0, vmax = threshold, cmap = colorrange)
        axs[0].tick_params(top = True, right = True)
        if force_correlation == True:
            ini = (frame-1)/frame_rate
            fin = frame/frame_rate
            try: axs[1].patches[0].remove()
            except: pass
            axs[1].axvspan(ini, fin, color = shade_color, alpha = shade_alpha, lw = 0, zorder=0)
            align_axis(ax = axs[1], align_to_ax = axs[0], axis = 'x')
            print(f'Sample Rate: {sample_rate} Hz')
            print(f'Downsampled Rate: {downsampled_rate} Hz')
        if showSaveButtons == True:
            add_button_savePNG(fig, folder, f'{name}_{channel}_frame{slider.value}', dpi,  transparent = transparentBack)
            add_button_savePDF(fig, folder, f'{name}_{channel}_frame{slider.value}')

        text = widgets.Text(value=f'showing frame:{frame}', disabled=True)
        display(text)

    slider = widgets.IntSlider(min=1, max=scan.num_frames, step=1, value=0)
    widgets.interact(update_frame, frame=slider)

    return fig
