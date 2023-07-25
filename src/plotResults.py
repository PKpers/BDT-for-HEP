#!/usr/bin/python
import sys
import ROOT
import numpy as np
ROOT.gROOT.SetBatch(True)

def set_axes_title(pltObj, xtitle, ytitle):
    pltObj.GetXaxis().SetTitle(str(xtitle))
    pltObj.GetYaxis().SetTitle(str(ytitle))
    return


def set_axes_range(pltObj, Xrange, Yrange):
    pltObj.GetXaxis().SetRangeUser(Xrange[0], Xrange[1]);
    pltObj.GetYaxis().SetRangeUser(Yrange[0], Yrange[1]);
    return

def set_markerstyle(pltObj, styling_opts):
    if not styling_opts or not any(styling_opts.values()) :
        return
    else:
        options = styling_opts.keys()
        if "mca" or "marker color attributes" in options:
            mca = styling_opts.get("mca")
            if not mca:
                mca = styling_opts.get("markercolor attribute")
            #
            pltObj.SetMarkerColorAlpha(mca[0], mca[1])
        #

        if "msz" in options or "markersize" in options:
            msz = styling_opts.get("msz")
            if not msz:
                msz = styling_opts.get("markersize")
            #
            pltObj.SetMarkerSize(msz)
        #

        if "ms" in options or "markerstyle" in options:
            ms = styling_opts.get("ms")
            if not ms:
                ms = styling_opts.get("markerstyle")
            #
            pltObj.SetMarkerStyle(ms)
        #

        if "fs" in options or "fillstyle" in options:
            fs = styling_opts.get("fs")
            if not fs:
                fs = styling_opts.get("fillstyle")
            #
            pltObj.SetFillStyle(fs)
        #

        if "fca" in options or "fill color attribute" in options:
            fca = styling_opts.get("fca")
            if not fca:
                fca = styling_opts.get("fill color attribute")
            #
            pltObj.SetFillColorAlpha(fca[0], fca[1])
        #
        return
#

def set_linestyle(pltObj, styling_opts):
    if not styling_opts or not any(styling_opts.values()) :
        return
    else:
        options = styling_opts.keys()
        if "lca" or "line color attributes" in options:
            lca = styling_opts.get("lca")
            if not lca:
                lca = styling_opts.get("line color attribute")
            #
            pltObj.SetLineColorAlpha(lca[0], lca[1])
        #

        if "ls" in options or "linestyle" in options:
            ls = styling_opts.get("ls")
            if not ls:
                ls = styling_opts.get("linestyle")
            #
            pltObj.SetLineStyle(ls)
        #

        if "fs" in options or "fillstyle" in options:
            fs = styling_opts.get("fs")
            if not fs:
                fs = styling_opts.get("fillstyle")
            #
            pltObj.SetFillStyle(fs)
        #

        if "fca" in options or "fill color attribute" in options:
            fca = styling_opts.get("fca")
            if not fca:
                fca = styling_opts.get("fill color attribute")
            #
            pltObj.SetFillColorAlpha(fca[0], fca[1])
        #
        return

def PlotHist(HistName, pltData, histOpts, axLabels, DrawOpts, **kwargs):
    df, data = pltData
    Nbins, Range = histOpts
    Xlabel, Ylabel = axLabels
    #
    myHist = df.Histo1D(
        (HistName, "", Nbins, Range[0], Range[1]),
        data
    )
    set_linestyle(myHist, kwargs)
    #myHist.SetLineColor(_Lcolor)
    set_axes_title(myHist, Xlabel, Ylabel)
    myHist.Draw(DrawOpts)
    return myHist

def create_legend(position, entries):
    xmin, ymin, xmax, ymax = position
    legend = ROOT.TLegend(xmin, ymin, xmax, ymax)
    legend.SetFillColor(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.03)
    for entry, attribs in entries.items():
        _name = entry
        _title, _linestyle = attribs
        legend.AddEntry(_name, _title, _linestyle)
    #
    legend.Draw('same')
    return legend
#
def add_Header(title):
    import ROOT
    label = ROOT.TLatex()
    label.SetTextSize(0.04)
    label.DrawLatexNDC(0.16, 0.92, "#bf{"+str(title)+"}")
    return

if __name__ == "__main__" :
    if len(sys.argv) != 3:
        print("usage: {} {} {}".format(sys.argv[0], "inp_file.root", "outfile.pdf"))
        exit(-1)

    # Configure input settings
    input_dir = "/home/kpapad/UG_thesis/Thesis/Bdt/out/Predictions/"
    treeName = "myTree"
    infile = sys.argv[1]
    
    # Configure output settings 
    output_dir = "/home/kpapad/UG_thesis/Thesis/Bdt/out/Plots/"
    outfile = sys.argv[2]

    if not outfile.endswith(".pdf"):
        outfile += ".pdf"

    output = output_dir+outfile

    # Load data
    df1 = ROOT.RDataFrame(treeName, input_dir+'{}test.root'.format(infile))
    df2= ROOT.RDataFrame(treeName, input_dir+'{}train.root'.format(infile))

    sigTest = df1.Filter("yTest_true == 1")
    sigTest_events = sigTest.Count().GetValue()

    bkgTest = df1.Filter("yTest_true == 0")
    bkgTest_events = bkgTest.Count().GetValue()

    sigTrain = df2.Filter("yRef_true == 1")
    sigTrain_events = sigTrain.Count().GetValue()

    bkgTrain = df2.Filter("yRef_true == 0")
    bkgTrain_events = bkgTrain.Count().GetValue()

    # Make the plots
    c = ROOT.TCanvas()
    c.cd()
    ROOT.gStyle.SetOptStat(0); ROOT.gStyle.SetTextFont(42)
    c.SaveAs(output+"[")
    #

    ## Plot the BDT histograms --------------------------------------------------------------------------------------------------
    data = [
        (sigTest, "sigT", "yTest_pred", "signal testing"),
        (sigTrain, "sigR", "yRef_pred", "signal training"),
        (bkgTest, "bkgT", "yTest_pred", "background testing"), 
        (bkgTrain, "bkgR", "yRef_pred", "background training"), 
    ]

    nbins = 50
    histRange = (0., 1)
    histOpts = (nbins, histRange)
    ax_labels = ("BDT score", "Events / Bin")
    Lcolor = (3, 4, 2, 1)
    legend_entries = {}
    h_ = []

    for i, d in enumerate(data):
        lca = [Lcolor[i], 1]
        if i == 0:
            draw_loc = 'E1'
            legend_mark = 'lep'
        elif i == 2:
            draw_loc = "E1same"
            legend_mark = "lep"
        elif i == 1 or i == 3:
            draw_loc = 'same'
            fs = [3004, None, 1001]
            fca = [(Lcolor[i], 1), None, (Lcolor[i], 0.15)]
            legend_mark = 'f'

        pltData = (d[0], d[2])
        hist = PlotHist(
            d[1], pltData, histOpts, ax_labels, draw_loc, 
            lca=lca, fca=fca[i-1], fs=fs[i-1] 
        )

        h_.append(hist)
        legend_entries[d[1]] = (d[-1], legend_mark)


    legend_loc = (0.6, 0.7, 0.7, 0.8)
    legend = create_legend(legend_loc, legend_entries)
    #add_Header('BDT histogram')

    ## Plot the BDT Histograms once without log scale 
    c.SetLogy(0)
    c.SaveAs(output)

    ## And once with using logscale
    c.SetLogy(1)
    c.SaveAs(output)

    # Calculate TPR and FPR and plot the roc curve -----------------------------------------------------------------
    c.Clear()
    c.SetCanvasSize(800, 800)
    ROOT.gPad.SetLeftMargin(0.11)
    ROOT.gPad.SetBottomMargin(0.11)

    integrals = []
    for h in h_ :
        h = h.GetValue()
        axis = h.GetXaxis()
        bmax = axis.FindBin(float(1))
        Int = [
            h.Integral(T, bmax)
            for T in range(h.GetNbinsX())
        ]
        integrals.append(Int)
        #
    #

    # Normalize the ROC curves
    TestingTPR,  TestingFPR = [
        np.array(integrals[0])/sigTest_events,
        np.array(integrals[2])/bkgTest_events
    ]

    TrainingTPR,  TrainingFPR = [
        np.array(integrals[1])/sigTrain_events,
        np.array(integrals[3])/bkgTrain_events 
    ]

    TestingTNR = 1 -TestingFPR
    TrainingTNR = 1 -TrainingFPR
    c.SetLogy(0)
    c.SetLogx(0)
    roc = ROOT.TMultiGraph('roc', '')
    roc_alt = ROOT.TMultiGraph('roc_alt', '')

    # Testing
    TestROC = ROOT.TGraph(len(TestingTPR), TestingFPR, TestingTPR)#(n,x,y)
    TestROC.SetName('TestROC')
    TestROC.SetTitle( 'Testing' )
    TestROC.SetLineColor(3)
    TestROC.SetLineWidth(5)
    TestROC.SetDrawOption( 'Al' )
    #
    TestROC_alt = ROOT.TGraph(len(TestingTPR), TestingTNR, TestingTPR)
    TestROC_alt.SetName('TestROC_alt')
    TestROC_alt.SetTitle( 'Testing' )
    TestROC_alt.SetLineColor(3)
    TestROC_alt.SetDrawOption( 'Al' )

    # Training
    TrainROC = ROOT.TGraph(len(TrainingTPR), TrainingFPR, TrainingTPR)
    TrainROC.SetName('TrainROC')
    TrainROC.SetTitle( 'Training' )
    TrainROC.SetLineColor(4)
    TrainROC.SetLineWidth(5)
    TrainROC.SetDrawOption( 'Al' )
    #
    TrainROC_alt = ROOT.TGraph(len(TrainingTPR), TrainingTNR, TrainingTPR)
    TrainROC_alt.SetName('TrainROC_alt')
    TrainROC_alt.SetTitle( 'Training' )
    TrainROC_alt.SetMarkerColor(4)
    TrainROC_alt.SetLineColor(4)
    TrainROC_alt.SetDrawOption( 'Al' )

    # Plot the curves
    roc.Add(TestROC)
    roc.Add(TrainROC)
    #

    set_axes_title(roc, "FPR", "TPR")
    roc.GetYaxis().SetLabelSize(0.054)
    roc.GetYaxis().SetTitleSize(0.054)
    roc.GetYaxis().SetTitleOffset(0.9)
    roc.GetXaxis().SetLabelSize(0.05)
    roc.GetXaxis().SetTitleSize(0.05)
    roc.Draw('AL')

    # legend
    legend = ROOT.TLegend(0.4, 0.4, 0.6, 0.6)
    legend.AddEntry(TestROC, 'Testing', 'lp')
    legend.AddEntry(TrainROC, 'Training', 'lp')
    legend.SetFillColor(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.057)
    legend.Draw('same')

    c.SaveAs(output)

    # Plot the roc curve with TNR instead of FPR
    roc_alt.Add(TestROC_alt)
    roc_alt.Add(TrainROC_alt)

    set_axes_title(roc_alt, "TNR", "TPR")
    roc_alt.Draw('ALP')

    legend = ROOT.TLegend(0.4, 0.4, 0.6, 0.6)
    legend.AddEntry(TestROC, 'Testing', 'lp')
    legend.AddEntry(TrainROC, 'Training', 'lp')
    legend.SetFillColor(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.03)
    legend.Draw('same')

    c.SaveAs(output)

    c.SaveAs(output+"]")

    ## You can add whatever else you want to plot here ##