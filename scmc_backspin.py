#!/usr/bin/env python3

# Copyright (c) 2015, Amit Zeisel, Gioele La Manno and Sten Linnarsson
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This .py file can be used as a library or a command-line version of BackSPIN,
# This version of BackSPIN was implemented by Gioele La Manno.
# The BackSPIN biclustering algorithm was developed by Amit Zeisel and is described
# in Zeisel et al. Cell types in the mouse cortex and hippocampus revealed by
# single-cell RNA-seq Science 2015 (PMID: 25700174, doi: 10.1126/science.aaa1934).
#
# Building using pyinstaller:
# pyinstaller -F backSPIN.py -n backspin-mac-64-bit
#

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy import *
import getopt
import sys
import os
import csv
from Cef_tools import CEF_obj
# import ipdb # debugging
import mypy
from scipy import stats

filename=str(sys.argv[1])


class Results:
    pass

def calc_loccenter(x, lin_log_flag):
    M,N = x.shape
    if N==1 and M>1:
        x = x.T
    M,N = x.shape
    loc_center = zeros(M)
    min_x = x.min(1)
    x = x - min_x[:,newaxis]
    for i in range(M):
        ind = where(x[i,:]>0)[0]
        if len(ind) != 0:
            if lin_log_flag == 1:
                w = x[i,ind]/sum(x[i,ind], 0)
            else:
                w = (2**x[i,ind])/sum(2**x[i,ind], 0)
            loc_center[i] = sum(w*ind, 0)
        else:
            loc_center[i] = 0

    return loc_center

def _calc_weights_matrix(mat_size, wid):
    '''Calculate Weight Matrix
    Parameters
    ----------
    mat_size: int
        dimension of the distance matrix
    wid: int
        parameter that controls the width of the neighbourood
    Returns
    -------
    weights_mat: 2-D array
        the weights matrix to multiply with the distance matrix

    '''
    #calculate square distance from the diagonal
    sqd = (arange(1,mat_size+1)[newaxis,:] - arange(1,mat_size+1)[:,newaxis])**2
    #make the distance relative to the mat_size
    norm_sqd = sqd/wid
    #evaluate a normal pdf
    weights_mat = exp(-norm_sqd/mat_size)
    #avoid useless precision that would slow down the matrix multiplication
    weights_mat -= 1e-6
    weights_mat[weights_mat<0] = 0
    #normalize row and column sum
    weights_mat /= sum(weights_mat,0)[newaxis,:]
    weights_mat /= sum(weights_mat,1)[:, newaxis]
    #fix asimmetries
    weights_mat = (weights_mat + weights_mat.T) / 2.
    return weights_mat


def _sort_neighbourhood( dist_matrix, wid ):
    '''Perform a single iteration of SPIN
    Parameters
    ----------
    dist_matrix: 2-D array
        distance matrix
    wid: int
        parameter that controls the width of the neighbourood
    Returns
    -------
    sorted_ind: 1-D array
        indexes that order the matrix

    '''
    assert wid > 0, 'Parameter wid < 0 is not allowed'
    mat_size = dist_matrix.shape[0]
    #assert mat_size>2, 'Matrix is too small to be sorted'
    weights_mat = _calc_weights_matrix(mat_size, wid)
    #Calculate the dot product (can be very slow for big mat_size)
    mismatch_score = dot(dist_matrix, weights_mat)
    energy, target_permutation = mismatch_score.min(1), mismatch_score.argmin(1)
    max_energy = max(energy)
    #Avoid points that have the same target_permutation value
    sort_score = target_permutation - 0.1 * sign( (mat_size/2 - target_permutation) ) * energy/max_energy
    #sort_score = target_permutation - 0.1 * sign( 1-2*(int(1000*energy/max_energy) % 2) ) * energy/max_energy # Alternative
    # Sorting the matrix
    sorted_ind = sort_score.argsort(0)[::-1]
    return sorted_ind


def sort_mat_by_neighborhood(dist_matrix, wid, times):
    '''Perform several iterations of SPIN using a fixed wid parameter
    Parameters
    ----------
    dist_matrix: 2-D array
        distance matrix
    wid: int
        parameter that controls the width of the neighbourood
    times: int
        number of repetitions
    verbose: bool
        print the progress
    Returns
    -------
    indexes: 1-D array
        indexes that order the matrix

    '''
    # original indexes
    indexes = arange(dist_matrix.shape[0])
    for i in range(times):
        #sort the sitance matrix according the previous iteration
        tmpmat = dist_matrix[indexes,:]
        tmpmat = tmpmat[:,indexes]
        sorted_ind = _sort_neighbourhood(tmpmat, wid);
        #resort the original indexes
        indexes = indexes[sorted_ind]
    return indexes


def _generate_widlist(data, axis=1, step=0.6):
    '''Generate a list of wid parameters to execute sort_mat_by_neighborhood
    Parameters
    ----------
    data: 2-D array
        the data matrix
    axis: int
        the axis to take in consideration
    step: float
        the increment between two successive wid parameters
    Returns
    -------
    wid_list: list of int
        list of wid parameters to run SPIN

    '''
    max_wid = data.shape[axis]*0.6
    new_wid = 1
    wid_list = []
    while new_wid < (1+step)*max_wid:
        wid_list.append( new_wid )
        new_wid = int(ceil( new_wid + new_wid*(step) +1))
    return wid_list[::-1]



def SPIN(dt, widlist=[10,1], iters=30, axis='both', verbose=False):
    """Run the original SPIN algorithm
    Parameters
    ----------
    dt: 2-D array
        the data matrix
    widlist: float or list of int
        If float is passed, it is used as step parameted of _generate_widlist,
        and widlist is generated to run SPIN.
        If list is passed it is used directly to run SPIN.
    iters: int
        number of repetitions for every wid in widlist
    axis: int
        the axis to take in consideration (must be 0, 1 or 'both')
    step: float
        the increment between two successive wid parameters
    Returns
    -------
    indexes: 1-D array (if axis in [0,1]) or tuple of 1-D array (if axis = 'both')
        indexes that sort the data matrix
    Notes
    -----
    Typical usage
    sorted_dt0 = SPIN(dt, iters=30, axis=0)
    sorted_dt1 = SPIN(dt, iters=30, axis=1)
    dt = dt[sorted_dt0,:]
    dt = dt[:,sorted_dt1]
    """
    IXc = arange(dt.shape[1])
    IXr = arange(dt.shape[0])
    assert axis in ['both', 0,1], 'axis must be 0, 1 or \'both\' '
    #Sort both axis
    if axis == 'both':
        CCc = 1 - corrcoef(dt.T)
        CCr = 1 - corrcoef(dt)
        if type(widlist) != list:
            widlist_r = _generate_widlist(dt, axis=0, step=widlist)
            widlist_c = _generate_widlist(dt, axis=1, step=widlist)
        if verbose:
                print('\nSorting genes.')
                print('Neighbourood=')
        for wid in widlist_r:
            if verbose:
                print(('%i, ' % wid))
                sys.stdout.flush()
            INDr = sort_mat_by_neighborhood(CCr, wid, iters)
            CCr = CCr[INDr,:][:,INDr]
            IXr = IXr[INDr]
        if verbose:
                print('\nSorting cells.')
                print('Neighbourood=')
        for wid in widlist_c:
            if verbose:
                print(('%i, ' % wid))
                sys.stdout.flush()
            INDc = sort_mat_by_neighborhood(CCc, wid, iters)
            CCc = CCc[:,INDc][INDc,:]
            IXc= IXc[INDc]
        return IXr, IXc
    #Sort rows
    elif axis == 0:
        CCr = 1 - corrcoef(dt)
        if type(widlist) != list:
            widlist = _generate_widlist(dt, axis=0, step=widlist)
        if verbose:
                print('\nSorting genes.\nNeighbourood=')
        for wid in widlist:
            if verbose:
                print('%i, ' % wid)
                sys.stdout.flush()
            INDr = sort_mat_by_neighborhood(CCr, wid, iters)
            CCr = CCr[INDr,:][:,INDr]
            IXr = IXr[INDr]
        return IXr
    #Sort columns
    elif axis == 1:
        CCc = 1 - corrcoef(dt.T)
        if type(widlist) != list:
            widlist = _generate_widlist(dt, axis=1, step=widlist)
        if verbose:
            print('\nSorting cells.\nNeighbourood=')
        for wid in widlist:
            if verbose:
                print('%i, ' % wid)
                sys.stdout.flush()
            INDc = sort_mat_by_neighborhood(CCc, wid, iters)
            CCc = CCc[:,INDc][INDc,:]
            IXc = IXc[INDc]
        return IXc


def backSPIN(data, numLevels=2, first_run_iters=10, first_run_step=0.05, runs_iters=8 ,runs_step=0.25,\
    split_limit_g=2, split_limit_c=2, stop_const = 1.15, low_thrs=0.2, verbose=False):
    '''Run the backSPIN algorithm
    Parameters
    ----------
    data: 2-D array
        the data matrix, rows should be genes and columns single cells/samples
    numLevels: int
        the number of splits that will be tried
    first_run_iters: float
        the iterations of the preparatory SPIN
    first_run_step: float
        the step parameter passed to _generate_widlist for the preparatory SPIN
    runs_iters: int
        the iterations parameter passed to the _divide_to_2and_resort.
        influences all the SPIN iterations except the first
    runs_step: float
        the step parameter passed to the _divide_to_2and_resort.
        influences all the SPIN iterations except the first
    wid: float
        the wid of every iteration of the splitting and resorting
    split_limit_g: int
        If the number of specific genes in a subgroup is smaller than this number
         splitting of that subgrup is not allowed
    split_limit_c: int
        If the number cells in a subgroup is smaller than this number splitting of
        that subgrup is not allowed
    stop_const: float
        minimum score that a breaking point has to reach to be suitable for splitting
    low_thrs: float
        genes with average lower than this threshold are assigned to either of the
        splitting group reling on genes that are higly correlated with them

    Returns
    -------
    results: Result object
        The results object contain the following attributes
        genes_order: 1-D array
            indexes (a permutation) sorting the genes
        cells_order: 1-D array
            indexes (a permutation) sorting the cells
        genes_gr_level: 2-D array
            for each depth level contains the cluster indexes for each gene
        cells_gr_level:
            for each depth level contains the cluster indexes for each cell
        cells_gr_level_sc:
            score of the splitting
        genes_bor_level:
            the border index between gene clusters
        cells_bor_level:
            the border index between cell clusters

    Notes
    -----
    Typical usage

    '''
    assert numLevels>0, '0 is not an available depth for backSPIN, use SPIN instead'
    #initialize some varaibles
    genes_bor_level = [[] for i in range(numLevels)]
    cells_bor_level = [[] for i in range(numLevels)]
    N,M = data.shape
    genes_order = arange(N)
    cells_order = arange(M)
    genes_gr_level = zeros((N,numLevels+1))
    cells_gr_level = zeros((M,numLevels+1))
    cells_gr_level_sc = zeros((M,numLevels+1))

    # Do a Preparatory SPIN on cells
    if verbose:
        print('\nPreparatory SPIN')
    ix1 = SPIN(data, widlist=_generate_widlist(data, axis=1, step=first_run_step), iters=first_run_iters, axis=1, verbose=verbose)
    cells_order = cells_order[ix1]

    #For every level of depth DO:
    for i in range(numLevels):
        k=0 # initialize group id counter
        # For every group generated at the parent level DO:
        for j in range( len( set(cells_gr_level[:,i]) ) ):
            # Extract the a data matrix of the genes at that level
            g_settmp = nonzero(genes_gr_level[:,i]==j)[0] #indexes of genes in the level j
            c_settmp = nonzero(cells_gr_level[:,i]==j)[0] #indexes of cells in the level j
            datatmp = data[ ix_(genes_order[g_settmp], cells_order[c_settmp]) ]
            # If we are not below the splitting limit for both genes and cells DO:
            if (len(g_settmp)>split_limit_g) & (len(c_settmp)>split_limit_c):
                # Split and SPINsort the two halves
                if i == numLevels-1:
                    divided = _divide_to_2and_resort(datatmp, wid=runs_step, iters_spin=runs_iters,\
                        stop_const=stop_const, low_thrs=low_thrs, sort_genes=True, verbose=verbose)
                else:
                    divided = _divide_to_2and_resort(datatmp, wid=runs_step, iters_spin=runs_iters,\
                        stop_const=stop_const, low_thrs=low_thrs, sort_genes=False,verbose=verbose)
                # _divide_to_2and_resort retruns an empty array in gr2 if the splitting condition was not satisfied
                if divided:
                    sorted_data_resort1, genes_resort1, cells_resort1,\
                    gr1, gr2, genesgr1, genesgr2, score1, score2 = divided
                    # Resort from the previous level
                    genes_order[g_settmp] = genes_order[g_settmp[genes_resort1]]
                    cells_order[c_settmp] = cells_order[c_settmp[cells_resort1]]
                    # Assign a numerical identifier to the groups
                    genes_gr_level[g_settmp[genesgr1],i+1] = k
                    genes_gr_level[g_settmp[genesgr2],i+1] = k+1
                    cells_gr_level[c_settmp[gr1],i+1] = k
                    cells_gr_level[c_settmp[gr2],i+1] = k+1
                    # Not really clear what sc is
                    cells_gr_level_sc[c_settmp[gr1],i+1] = score1
                    cells_gr_level_sc[c_settmp[gr2],i+1] = score2
                    # Augment the counter of 2 becouse two groups were generated from one
                    k = k+2
                else:
                    # The split is not convenient, keep everithing the same
                    genes_gr_level[g_settmp,i+1] = k
                    # if it is the deepest level: perform gene sorting
                    if i == numLevels-1:
                        if (datatmp.shape[0] > 2 )and (datatmp.shape[1] > 2):
                            genes_resort1 = SPIN(datatmp, widlist=runs_step, iters=runs_iters, axis=0, verbose=verbose)
                            genes_order[g_settmp] = genes_order[g_settmp[genes_resort1]]
                    cells_gr_level[c_settmp,i+1] = k
                    cells_gr_level_sc[c_settmp,i+1] = cells_gr_level_sc[c_settmp,i]
                    # Augment of 1 becouse no new group was generated
                    k = k+1
            else:
                # Below the splitting limit: the split is not convenient, keep everithing the same
                genes_gr_level[g_settmp,i+1] = k
                cells_gr_level[c_settmp,i+1] = k
                cells_gr_level_sc[c_settmp,i+1] = cells_gr_level_sc[c_settmp,i]
                # Augment of 1 becouse no new group was generated
                k = k+1

        # Find boundaries
        genes_bor_level[i] = r_[0, nonzero(diff(genes_gr_level[:,i+1])>0)[0]+1, data.shape[0] ]
        cells_bor_level[i] = r_[0, nonzero(diff(cells_gr_level[:,i+1])>0)[0]+1, data.shape[1] ]

    #dataout_sorted = data[ ix_(genes_order,cells_order) ]

    results = Results()
    results.genes_order = genes_order
    results.cells_order = cells_order
    results.genes_gr_level = genes_gr_level
    results.cells_gr_level = cells_gr_level
    results.cells_gr_level_sc = cells_gr_level_sc
    results.genes_bor_level = genes_bor_level
    results.cells_bor_level = cells_bor_level

    return results



def _divide_to_2and_resort(sorted_data, wid, iters_spin=8, stop_const = 1.15, low_thrs=0.2 , sort_genes=True, verbose=False):
    '''Core function of backSPIN: split the datamatrix in two and resort the two halves

    Parameters
    ----------
    sorted_data: 2-D array
        the data matrix, rows should be genes and columns single cells/samples
    wid: float
        wid parameter to give to widlist parameter of th SPIN fucntion
    stop_const: float
        minimum score that a breaking point has to reach to be suitable for splitting
    low_thrs: float
        if the difference between the average expression of two groups is lower than threshold the algorythm
        uses higly correlated gens to assign the gene to one of the two groups
    verbose: bool
        information about the split is printed

    Returns
    -------
    '''

    # Calculate correlation matrix for cells and genes
    Rcells = corrcoef(sorted_data.T)
    Rgenes = corrcoef(sorted_data)
    # Look for the optimal breaking point
    N = Rcells.shape[0]
    score = zeros(N)
    for i in range(2,N-2):
        if i == 2:
            tmp1 = sum( Rcells[:i,:i] )
            tmp2 = sum( Rcells[i:,i:] )
            score[i] = (tmp1+tmp2) / float(i**2 + (N-i)**2)
        else:
            tmp1 += sum(Rcells[i-1,:i]) + sum(Rcells[:i-1,i-1]);
            tmp2 -= sum(Rcells[i-1:,i-1]) + sum(Rcells[i-1,i:]);
            score[i] = (tmp1+tmp2) / float(i**2 + (N-i)**2)

    breakp1 = argmax(score)
    score1 = Rcells[:breakp1,:breakp1]
    score1 = triu(score1)
    score1 = mean( score1[score1 != 0] )
    score2 = Rcells[breakp1:,breakp1:]
    score2 = triu(score2)
    score2 = mean( score2[score2 != 0] )
    avg_tot = triu(Rcells)
    avg_tot = mean( avg_tot[avg_tot != 0] )

    # If it is convenient to break
    if (max([score1,score2])/avg_tot) > stop_const:
        # Divide in two groups
        gr1 = arange(N)[:breakp1]
        gr2 = arange(N)[breakp1:]
        # and assign the genes into the two groups
        mean_gr1 = sorted_data[:,gr1].mean(1)
        mean_gr2 = sorted_data[:,gr2].mean(1)
        concat_loccenter_gr1 = c_[ calc_loccenter(sorted_data[:,gr1], 2), calc_loccenter(sorted_data[:,gr1][...,::-1], 2) ]
        concat_loccenter_gr2 = c_[ calc_loccenter(sorted_data[:,gr2], 2), calc_loccenter(sorted_data[:,gr2][...,::-1], 2) ]
        center_gr1, flip_flag1 = concat_loccenter_gr1.min(1), concat_loccenter_gr1.argmin(1)
        center_gr2, flip_flag2 = concat_loccenter_gr2.max(1), concat_loccenter_gr2.argmax(1)
        sorted_data_tmp = array( sorted_data )
        sorted_data_tmp[ix_(flip_flag1==1,gr1)] = sorted_data[ix_(flip_flag1==1,gr1)][...,::-1]
        sorted_data_tmp[ix_(flip_flag2==1,gr2)] = sorted_data[ix_(flip_flag2==1,gr2)][...,::-1]
        loc_center = calc_loccenter(sorted_data_tmp, 2)

        imax = zeros(loc_center.shape)
        imax[loc_center<=breakp1] = 1
        imax[loc_center>breakp1] = 2

        genesgr1 = where(imax==1)[0]
        genesgr2 = where(imax==2)[0]
        if size(genesgr1) == 0:
            IN = argmax(mean_gr1)
            genesgr1 = array([IN])
            genesgr2 = setdiff1d(genesgr2, IN)
        elif size(genesgr2) == 0:
            IN = argmax(mean_gr2)
            genesgr2 = array([IN])
            genesgr1 = setdiff1d(genesgr1, IN)

        if verbose:
            print('\nSplitting (%i, %i) ' %  sorted_data.shape)
            print('in (%i,%i) ' % (genesgr1.shape[0],gr1.shape[0]))
            print('and (%i,%i)' % (genesgr2.shape[0],gr2.shape[0]))
            sys.stdout.flush()

        # Data of group1
        datagr1 = sorted_data[ix_(genesgr1,gr1)]
        # zero center
        datagr1 = datagr1 - datagr1.mean(1)[:,newaxis]
        # Resort group1
        if min( datagr1.shape ) > 1:
            if sort_genes:
                genesorder1,cellorder1 = SPIN(datagr1, widlist=wid, iters=iters_spin, axis='both', verbose=verbose)
            else:
                cellorder1 = SPIN(datagr1, widlist=wid, iters=iters_spin, axis=1, verbose=verbose)
                genesorder1 = arange(datagr1.shape[0])
        elif len(genesgr1) == 1:
            genesorder1 = 0
            cellorder1 = argsort( datagr1[0,:] )
        elif len(gr1) == 1:
            cellorder1 = 0
            genesorder1 = argsort( datagr1[:,0] )

        # Data of group2
        datagr2 = sorted_data[ix_(genesgr2,gr2)]
        # zero center
        datagr2 = datagr2 - datagr2.mean(1)[:,newaxis]
        # Resort group2
        if min( datagr2.shape )>1:
            if sort_genes:
                genesorder2, cellorder2 = SPIN(datagr2, widlist=wid, iters=iters_spin, axis='both',verbose=verbose)
            else:
                cellorder2 = SPIN(datagr2, widlist=wid, iters=iters_spin, axis=1,verbose=verbose)
                genesorder2 = arange(datagr2.shape[0])
        elif len(genesgr2) == 1:
            genesorder2 = 0
            cellorder2 = argsort(datagr2[0,:])
        elif len(gr2) == 1:
            cellorder2 = 0
            genesorder2 = argsort(datagr2[:,0])

        # contcatenate cells and genes indexes
        genes_resort1 = r_[genesgr1[genesorder1], genesgr2[genesorder2] ]
        cells_resort1 = r_[gr1[cellorder1], gr2[cellorder2] ]
        genesgr1 = arange(len(genesgr1))
        genesgr2 = arange(len(genesgr1), len(sorted_data[:,0]))
        # resort
        sorted_data_resort1 = sorted_data[ix_(genes_resort1,cells_resort1)]

        return sorted_data_resort1, genes_resort1, cells_resort1, gr1, gr2, genesgr1, genesgr2, score1, score2

    else:
        if verbose:
            print('Low splitting score was : %.4f' % (max([score1,score2])/avg_tot))
        return False


def fit_CV(mu, cv, fit_method='Exp', svr_gamma=0.06, x0=[0.5,0.5], verbose=False):
    '''Fits a noise model (CV vs mean)
    Parameters
    ----------
    mu: 1-D array
        mean of the genes (raw counts)
    cv: 1-D array
        coefficient of variation for each gene
    fit_method: string
        allowed: 'SVR', 'Exp', 'binSVR', 'binExp'
        default: 'SVR'(requires scikit learn)
        SVR: uses Support vector regression to fit the noise model
        Exp: Parametric fit to cv = mu^(-a) + b
        bin: before fitting the distribution of mean is normalized to be
             uniform by downsampling and resampling.
    Returns
    -------
    score: 1-D array
        Score is the relative position with respect of the fitted curve
    mu_linspace: 1-D array
        x coordiantes to plot (min(log2(mu)) -> max(log2(mu)))
    cv_fit: 1-D array
        y=f(x) coordinates to plot
    pars: tuple or None

    '''
    log2_m = log2(mu)
    log2_cv = log2(cv)

    if len(mu)>1000 and 'bin' in fit_method:
        #histogram with 30 bins
        n,xi = histogram(log2_m,30)
        med_n = percentile(n,50)
        for i in range(0,len(n)):
            # index of genes within the ith bin
            ind = where( (log2_m >= xi[i]) & (log2_m < xi[i+1]) )[0]
            if len(ind)>med_n:
                #Downsample if count is more than median
                ind = ind[random.permutation(len(ind))]
                ind = ind[:len(ind)-med_n]
                mask = ones(len(log2_m), dtype=bool)
                mask[ind] = False
                log2_m = log2_m[mask]
                log2_cv = log2_cv[mask]
            elif (around(med_n/len(ind))>1) and (len(ind)>5):
                #Duplicate if count is less than median
                log2_m = r_[ log2_m, tile(log2_m[ind], around(med_n/len(ind))-1) ]
                log2_cv = r_[ log2_cv, tile(log2_cv[ind], around(med_n/len(ind))-1) ]
    else:
        if 'bin' in fit_method:
            print('More than 1000 input feature needed for bin correction.')
        pass

    if 'SVR' in fit_method:
        try:
            from sklearn.svm import SVR
            if svr_gamma == 'auto':
                svr_gamma = 1000./len(mu)
            #Fit the Support Vector Regression
            clf = SVR(gamma=svr_gamma)
            clf.fit(log2_m[:,newaxis], log2_cv)
            fitted_fun = clf.predict
            score = log2(cv) - fitted_fun(log2(mu)[:,newaxis])
            params = None
            #The coordinates of the fitted curve
            mu_linspace = linspace(min(log2_m),max(log2_m))
            cv_fit = fitted_fun(mu_linspace[:,newaxis])
            return score, mu_linspace, cv_fit , params

        except ImportError:
            if verbose:
                print('SVR fit requires scikit-learn python library. Using exponential instead.')
            if 'bin' in fit_method:
                return fit_CV(mu, cv, fit_method='binExp', x0=x0)
            else:
                return fit_CV(mu, cv, fit_method='Exp', x0=x0)
    elif 'Exp' in fit_method:
        from scipy.optimize import minimize
        #Define the objective function to fit (least squares)
        fun = lambda x, log2_m, log2_cv: sum(abs( log2( (2.**log2_m)**(-x[0])+x[1]) - log2_cv ))
        #Fit using Nelder-Mead algorythm
        optimization =  minimize(fun, x0, args=(log2_m,log2_cv), method='Nelder-Mead')
        params = optimization.x
        #The fitted function
        fitted_fun = lambda log_mu: log2( (2.**log_mu)**(-params[0]) + params[1])
        # Score is the relative position with respect of the fitted curve
        score = log2(cv) - fitted_fun(log2(mu))
        #The coordinates of the fitted curve
        mu_linspace = linspace(min(log2_m),max(log2_m))
        cv_fit = fitted_fun(mu_linspace)
        return score, mu_linspace, cv_fit , params



def feature_selection(data,thrs, verbose=False):
    if thrs>= data.shape[0]:
        if verbose:
            print("Trying to select %i features but only %i genes available." %( thrs, data.shape[0]))
            print("Skipping feature selection")
        return arange(data.shape[0])
    ix_genes = arange(data.shape[0])
    threeperK = int(ceil(3*data.shape[1]/1000.))
    zerotwoperK = int(floor(0.3*data.shape[1]/1000.))
    # is at least 1 molecule in 0.3% of thecells, is at least 2 molecules in 0.03% of the cells
    condition = (sum(data>=1, 1)>= threeperK) & (sum(data>=2, 1)>=zerotwoperK)
    ix_genes = ix_genes[condition]

    mu = data[ix_genes,:].mean(1)
    sigma = data[ix_genes,:].std(1, ddof=1)
    cv = sigma/mu

    try:
        score, mu_linspace, cv_fit , params = fit_CV(mu,cv,fit_method='SVR', verbose=verbose)
    except ImportError:
        print("WARNING: Feature selection was skipped becouse scipy is required. Install scipy to run feature selection.")
        return arange(data.shape[0])

    return ix_genes[argsort(score)[::-1]][:thrs]

def usage_quick():

    message ='''usage: backSPIN [-hbv] [-i inputfile] [-o outputfolder] [-d int] [-f int] [-t int] [-s float] [-T int] [-S float] [-g int] [-c int] [-k float] [-r float]
    manual: backSPIN -h
    '''
    print(message)

def usage():

    message='''
       backSPIN commandline tool
       -------------------------

       The options are as follows:

       -i [inputfile]
       --input=[inputfile]
              Path of the cef formatted tab delimited file.
              Rows should be genes and columns single cells/samples.
              For further information on the cef format visit:
              https://github.com/linnarsson-lab/ceftools

       -o [outputfile]
       --output=[outputfile]
              The name of the file to which the output will be written

       -d [int]
              Depth/Number of levels: The number of nested splits that will be tried by the algorithm
       -t [int]
              Number of the iterations used in the preparatory SPIN.
              Defaults to 10
       -f [int]
              Feature selection is performed before BackSPIN. Argument controls how many genes are seleceted.
              Selection is based on expected noise (a curve fit to the CV-vs-mean plot).
       -s [float]
              Controls the decrease rate of the width parameter used in the preparatory SPIN.
              Smaller values will increase the number of SPIN iterations and result in higher
              precision in the first step but longer execution time.
              Defaults to 0.1
       -T [int]
              Number of the iterations used for every width parameter.
              Does not apply on the first run (use -t instead)
              Defaults to 8
       -S [float]
              Controls the decrease rate of the width parameter.
              Smaller values will increase the number of SPIN iterations and result in higher
              precision but longer execution time.
              Does not apply on the first run (use -s instead)
              Defaults to 0.3
       -g [int]
              Minimal number of genes that a group must contain for splitting to be allowed.
              Defaults to 2
       -c [int]
              Minimal number of cells that a group must contain for splitting to be allowed.
              Defaults to 2
       -k [float]
              Minimum score that a breaking point has to reach to be suitable for splitting.
              Defaults to 1.15
       -r [float]
              If the difference between the average expression of two groups is lower than threshold the algorythm
              uses higly correlated genes to assign the gene to one of the two groups
              Defaults to 0.2
       -b [axisvalue]
              Run normal SPIN instead of backSPIN.
              Normal spin accepts the parameters -T -S
              An axis value 0 to only sort genes (rows), 1 to only sort cells (columns) or 'both' for both
              must be passed
       -v
              Verbose. Print  to the stdoutput extra details of what is happening

    '''

    print(message)


input_path = None
outfiles_path = None
numLevels=2 # -d
feature_fit = False # -f
feature_genes = 2000
first_run_iters=10 # -t
first_run_step=0.1 # -s
runs_iters=8 # -T
runs_step=0.3 # -S
split_limit_g=2 # -g
split_limit_c=2 # -c
stop_const = 1.15 # -k
low_thrs=0.2 # -r
normal_spin = False #-b
normal_spin_axis = 'both'
verbose=True # -v


df = pd.read_csv(filename, sep="\t")
df = df.filter(regex='_mcc$')
# df = df.filter(regex='nuclei')

all_levels = [[df.columns.tolist()]]

# EAM -- commenting the line below. If uncommented it would predefine the first split
# all_levels.append([['Pool_606_AD006_indexed_R1_mcc','Pool_616_AD006_indexed_R1_mcc','Pool_597_AD006_indexed_R1_mcc','nuclei_587_mcc','Pool_745_AD008_indexed_R1_mcc','Pool_709_AD010_indexed_R1_mcc','Pool_987_AD002_indexed_R1_mcc','Pool_889_AD010_indexed_R1_mcc','Pool_806_AD002_indexed_R1_mcc','Pool_744_AD010_indexed_R1_mcc','Pool_8_AD006_indexed_R1_mcc','Pool_873_AD010_indexed_R1_mcc','nuclei_311_mcc','Pool_796_AD006_indexed_R1_mcc','nuclei_565_mcc','Pool_1254_AD008_indexed_R1_mcc','Pool_1196_AD002_indexed_R1_mcc','Pool_1103_AD010_indexed_R1_mcc','Pool_680_AD008_indexed_R1_mcc','Pool_747_AD010_indexed_R1_mcc','Pool_684_AD008_indexed_R1_mcc','Pool_887_AD010_indexed_R1_mcc','Pool_823_AD006_indexed_R1_mcc','Pool_1128_AD006_indexed_R1_mcc','Pool_1069_AD010_indexed_R1_mcc','Pool_880_AD008_indexed_R1_mcc','Pool_998_AD006_indexed_R1_mcc','Pool_968_AD006_indexed_R1_mcc','Pool_1216_AD010_indexed_R1_mcc','Pool_805_AD002_indexed_R1_mcc','Pool_1043_AD008_indexed_R1_mcc','Pool_1004_AD006_indexed_R1_mcc','Pool_933_AD006_indexed_R1_mcc','Pool_961_AD006_indexed_R1_mcc','Pool_767_AD008_indexed_R1_mcc','Pool_1003_AD002_indexed_R1_mcc','Pool_909_AD010_indexed_R1_mcc','Pool_915_AD006_indexed_R1_mcc','Pool_763_AD010_indexed_R1_mcc','Pool_694_AD008_indexed_R1_mcc','Pool_1098_AD010_indexed_R1_mcc','Pool_721_AD010_indexed_R1_mcc','Pool_1092_AD010_indexed_R1_mcc','nuclei_378_mcc','Pool_1097_AD008_indexed_R1_mcc','Pool_1275_AD010_indexed_R1_mcc','Pool_1077_AD010_indexed_R1_mcc','Pool_841_AD010_indexed_R1_mcc','Pool_1077_AD008_indexed_R1_mcc','Pool_670_AD006_indexed_R1_mcc','Pool_883_AD010_indexed_R1_mcc','Pool_1029_AD010_indexed_R1_mcc','Pool_918_AD006_indexed_R1_mcc','Pool_1007_AD006_indexed_R1_mcc','Pool_655_AD006_indexed_R1_mcc','Pool_736_AD010_indexed_R1_mcc','Pool_649_AD002_indexed_R1_mcc','Pool_1123_AD006_indexed_R1_mcc','Pool_988_AD002_indexed_R1_mcc','Pool_812_AD006_indexed_R1_mcc','Pool_907_AD010_indexed_R1_mcc','Pool_867_AD010_indexed_R1_mcc','Pool_1041_AD008_indexed_R1_mcc','Pool_829_AD002_indexed_R1_mcc','Pool_732_AD008_indexed_R1_mcc','Pool_834_AD006_indexed_R1_mcc','Pool_1201_AD008_indexed_R1_mcc','Pool_797_AD002_indexed_R1_mcc','nuclei_273_mcc','nuclei_276_mcc','nuclei_348_mcc','Pool_1014_AD010_indexed_R1_mcc','Pool_1286_AD010_indexed_R1_mcc','Pool_852_AD010_indexed_R1_mcc','Pool_1070_AD010_indexed_R1_mcc','Pool_589_AD006_indexed_R1_mcc','Pool_884_AD008_indexed_R1_mcc','Pool_754_AD010_indexed_R1_mcc','Pool_1001_AD002_indexed_R1_mcc','Pool_1296_AD010_indexed_R1_mcc','Pool_1274_AD008_indexed_R1_mcc','Pool_1109_AD006_indexed_R1_mcc','Pool_700_AD010_indexed_R1_mcc','Pool_1294_AD002_indexed_R1_mcc','Pool_1236_AD010_indexed_R1_mcc','Pool_1170_AD002_indexed_R1_mcc','Pool_706_AD010_indexed_R1_mcc','nuclei_341_mcc','nuclei_552_mcc','Pool_780_AD006_indexed_R1_mcc','Pool_819_AD002_indexed_R1_mcc','Pool_809_AD006_indexed_R1_mcc','Pool_627_AD006_indexed_R1_mcc','Pool_1228_AD008_indexed_R1_mcc','Pool_886_AD010_indexed_R1_mcc','Pool_745_AD010_indexed_R1_mcc','Pool_641_AD002_indexed_R1_mcc','Pool_1122_AD002_indexed_R1_mcc','nuclei_583_mcc','Pool_705_AD010_indexed_R1_mcc','Pool_975_AD002_indexed_R1_mcc','Pool_874_AD008_indexed_R1_mcc','Pool_950_AD006_indexed_R1_mcc','Pool_268_AD006_indexed_R1_mcc','Pool_737_AD008_indexed_R1_mcc','Pool_908_AD010_indexed_R1_mcc','Pool_1240_AD010_indexed_R1_mcc','Pool_906_AD008_indexed_R1_mcc','Pool_1141_AD002_indexed_R1_mcc','Pool_1105_AD006_indexed_R1_mcc','Pool_1012_AD010_indexed_R1_mcc','Pool_1281_AD008_indexed_R1_mcc','Pool_912_AD008_indexed_R1_mcc','Pool_821_AD006_indexed_R1_mcc','Pool_1107_AD006_indexed_R1_mcc','Pool_666_AD002_indexed_R1_mcc','Pool_1056_AD010_indexed_R1_mcc','Pool_1026_AD010_indexed_R1_mcc','Pool_806_AD006_indexed_R1_mcc','Pool_685_AD010_indexed_R1_mcc','Pool_1044_AD010_indexed_R1_mcc','Pool_782_AD006_indexed_R1_mcc','Pool_723_AD010_indexed_R1_mcc','Pool_625_AD002_indexed_R1_mcc','Pool_678_AD010_indexed_R1_mcc','Pool_1062_AD010_indexed_R1_mcc','nuclei_274_mcc','Pool_957_AD006_indexed_R1_mcc','Pool_920_AD006_indexed_R1_mcc','Pool_1079_AD008_indexed_R1_mcc','Pool_1253_AD008_indexed_R1_mcc','Pool_642_AD002_indexed_R1_mcc','Pool_969_AD002_indexed_R1_mcc','Pool_932_AD006_indexed_R1_mcc','Pool_762_AD010_indexed_R1_mcc','Pool_831_AD006_indexed_R1_mcc','Pool_64_AD002_indexed_R1_mcc','Pool_728_AD008_indexed_R1_mcc','Pool_761_AD008_indexed_R1_mcc','Pool_1260_AD008_indexed_R1_mcc','Pool_1130_AD006_indexed_R1_mcc','Pool_1022_AD008_indexed_R1_mcc','Pool_957_AD002_indexed_R1_mcc','Pool_1268_AD010_indexed_R1_mcc','Pool_1295_AD010_indexed_R1_mcc','Pool_589_AD002_indexed_R1_mcc','Pool_774_AD006_indexed_R1_mcc','Pool_1100_AD010_indexed_R1_mcc','Pool_915_AD002_indexed_R1_mcc','Pool_1110_AD006_indexed_R1_mcc','Pool_1146_AD002_indexed_R1_mcc','Pool_666_AD006_indexed_R1_mcc','Pool_1118_AD006_indexed_R1_mcc','nuclei_328_mcc','Pool_657_AD006_indexed_R1_mcc','Pool_1061_AD008_indexed_R1_mcc','Pool_982_AD006_indexed_R1_mcc','Pool_1270_AD008_indexed_R1_mcc','Pool_1214_AD008_indexed_R1_mcc','Pool_590_AD006_indexed_R1_mcc','Pool_1112_AD006_indexed_R1_mcc','Pool_610_AD006_indexed_R1_mcc','nuclei_323_mcc','Pool_904_AD008_indexed_R1_mcc','Pool_1127_AD002_indexed_R1_mcc','Pool_1256_AD008_indexed_R1_mcc','Pool_1015_AD008_indexed_R1_mcc','Pool_730_AD010_indexed_R1_mcc','Pool_990_AD006_indexed_R1_mcc','Pool_1211_AD010_indexed_R1_mcc','Pool_651_AD002_indexed_R1_mcc','Pool_694_AD010_indexed_R1_mcc','Pool_1112_AD002_indexed_R1_mcc','Pool_877_AD008_indexed_R1_mcc','Pool_835_AD006_indexed_R1_mcc','Pool_629_AD002_indexed_R1_mcc','Pool_798_AD002_indexed_R1_mcc','Pool_759_AD010_indexed_R1_mcc','Pool_1030_AD008_indexed_R1_mcc','Pool_1068_AD010_indexed_R1_mcc','Pool_1187_AD006_indexed_R1_mcc','Pool_1019_AD010_indexed_R1_mcc','Pool_1276_AD008_indexed_R1_mcc','Pool_724_AD010_indexed_R1_mcc','Pool_1046_AD008_indexed_R1_mcc','Pool_837_AD002_indexed_R1_mcc','Pool_652_AD002_indexed_R1_mcc','Pool_703_AD008_indexed_R1_mcc','Pool_1053_AD008_indexed_R1_mcc','Pool_742_AD010_indexed_R1_mcc','Pool_753_AD010_indexed_R1_mcc','Pool_1027_AD010_indexed_R1_mcc','Pool_1179_AD006_indexed_R1_mcc','Pool_1157_AD002_indexed_R1_mcc','Pool_365_AD010_indexed_R1_mcc','Pool_976_AD002_indexed_R1_mcc','Pool_720_AD008_indexed_R1_mcc','Pool_964_AD006_indexed_R1_mcc','nuclei_334_mcc','Pool_1100_AD008_indexed_R1_mcc','Pool_600_AD006_indexed_R1_mcc','Pool_766_AD008_indexed_R1_mcc','Pool_1102_AD008_indexed_R1_mcc','Pool_1047_AD008_indexed_R1_mcc','Pool_720_AD010_indexed_R1_mcc','Pool_678_AD008_indexed_R1_mcc','Pool_853_AD010_indexed_R1_mcc','Pool_1175_AD002_indexed_R1_mcc','Pool_650_AD006_indexed_R1_mcc','Pool_905_AD010_indexed_R1_mcc','Pool_989_AD006_indexed_R1_mcc','Pool_613_AD002_indexed_R1_mcc','Pool_1036_AD010_indexed_R1_mcc','Pool_1181_AD002_indexed_R1_mcc','Pool_1257_AD010_indexed_R1_mcc','Pool_986_AD002_indexed_R1_mcc','Pool_1146_AD006_indexed_R1_mcc','Pool_640_AD006_indexed_R1_mcc','Pool_703_AD010_indexed_R1_mcc','Pool_1189_AD002_indexed_R1_mcc','Pool_1144_AD006_indexed_R1_mcc','Pool_1090_AD010_indexed_R1_mcc','Pool_1155_AD006_indexed_R1_mcc','Pool_660_AD002_indexed_R1_mcc','Pool_698_AD008_indexed_R1_mcc','nuclei_570_mcc','Pool_751_AD008_indexed_R1_mcc','Pool_1262_AD010_indexed_R1_mcc','Pool_611_AD006_indexed_R1_mcc','Pool_1063_AD010_indexed_R1_mcc','nuclei_307_mcc','Pool_896_AD008_indexed_R1_mcc','Pool_1292_AD008_indexed_R1_mcc','Pool_942_AD002_indexed_R1_mcc','Pool_657_AD002_indexed_R1_mcc','Pool_1219_AD010_indexed_R1_mcc','Pool_879_AD010_indexed_R1_mcc','Pool_1283_AD008_indexed_R1_mcc','Pool_1227_AD002_indexed_R1_mcc','Pool_801_AD006_indexed_R1_mcc','nuclei_584_mcc','Pool_1145_AD002_indexed_R1_mcc','Pool_603_AD006_indexed_R1_mcc','Pool_1152_AD006_indexed_R1_mcc','Pool_1120_AD006_indexed_R1_mcc','Pool_807_AD006_indexed_R1_mcc','Pool_669_AD002_indexed_R1_mcc','nuclei_346_mcc','Pool_1246_AD008_indexed_R1_mcc','Pool_1141_AD006_indexed_R1_mcc','Pool_1045_AD010_indexed_R1_mcc','Pool_996_AD002_indexed_R1_mcc','Pool_919_AD006_indexed_R1_mcc','Pool_883_AD008_indexed_R1_mcc','Pool_1234_AD010_indexed_R1_mcc','Pool_1249_AD008_indexed_R1_mcc','Pool_762_AD008_indexed_R1_mcc','Pool_1261_AD008_indexed_R1_mcc','Pool_1161_AD006_indexed_R1_mcc','Pool_858_AD010_indexed_R1_mcc','Pool_1138_AD002_indexed_R1_mcc','Pool_1096_AD010_indexed_R1_mcc','Pool_1067_AD010_indexed_R1_mcc','Pool_937_AD006_indexed_R1_mcc','Pool_679_AD008_indexed_R1_mcc','Pool_630_AD002_indexed_R1_mcc','Pool_1241_AD008_indexed_R1_mcc','Pool_1009_AD010_indexed_R1_mcc','Pool_803_AD006_indexed_R1_mcc','Pool_1051_AD010_indexed_R1_mcc','Pool_750_AD008_indexed_R1_mcc','Pool_947_AD002_indexed_R1_mcc','Pool_940_AD002_indexed_R1_mcc','Pool_850_AD010_indexed_R1_mcc','Pool_1215_AD008_indexed_R1_mcc','Pool_1142_AD006_indexed_R1_mcc','Pool_758_AD008_indexed_R1_mcc','Pool_939_AD002_indexed_R1_mcc','Pool_870_AD008_indexed_R1_mcc','Pool_908_AD008_indexed_R1_mcc','Pool_676_AD008_indexed_R1_mcc','Pool_916_AD006_indexed_R1_mcc','Pool_1156_AD002_indexed_R1_mcc','nuclei_541_mcc','Pool_967_AD006_indexed_R1_mcc','Pool_1210_AD008_indexed_R1_mcc','Pool_1195_AD006_indexed_R1_mcc','nuclei_272_mcc','Pool_1219_AD008_indexed_R1_mcc','Pool_843_AD008_indexed_R1_mcc','Pool_749_AD008_indexed_R1_mcc','Pool_1231_AD008_indexed_R1_mcc','Pool_931_AD006_indexed_R1_mcc','Pool_919_AD002_indexed_R1_mcc','Pool_1013_AD010_indexed_R1_mcc','Pool_899_AD010_indexed_R1_mcc','Pool_1171_AD006_indexed_R1_mcc','Pool_683_AD008_indexed_R1_mcc','Pool_696_AD010_indexed_R1_mcc','Pool_122_AD002_indexed_R1_mcc','Pool_947_AD006_indexed_R1_mcc','nuclei_596_mcc','Pool_1064_AD008_indexed_R1_mcc','Pool_765_AD008_indexed_R1_mcc','Pool_1187_AD002_indexed_R1_mcc','Pool_724_AD008_indexed_R1_mcc','Pool_924_AD002_indexed_R1_mcc','Pool_689_AD008_indexed_R1_mcc','Pool_1172_AD006_indexed_R1_mcc','Pool_1229_AD010_indexed_R1_mcc','Pool_783_AD002_indexed_R1_mcc','Pool_872_AD008_indexed_R1_mcc','Pool_936_AD006_indexed_R1_mcc','Pool_1096_AD008_indexed_R1_mcc','Pool_953_AD002_indexed_R1_mcc','Pool_660_AD006_indexed_R1_mcc','Pool_643_AD006_indexed_R1_mcc','Pool_1101_AD008_indexed_R1_mcc','Pool_1186_AD002_indexed_R1_mcc','Pool_1227_AD008_indexed_R1_mcc','Pool_1006_AD006_indexed_R1_mcc','nuclei_428_mcc','Pool_595_AD006_indexed_R1_mcc','Pool_153_AD006_indexed_R1_mcc','Pool_774_AD002_indexed_R1_mcc','nuclei_304_mcc','Pool_1179_AD002_indexed_R1_mcc','Pool_767_AD010_indexed_R1_mcc','Pool_733_AD010_indexed_R1_mcc','nuclei_352_mcc','Pool_1076_AD008_indexed_R1_mcc','Pool_995_AD006_indexed_R1_mcc','Pool_601_AD006_indexed_R1_mcc','Pool_929_AD006_indexed_R1_mcc','Pool_729_AD008_indexed_R1_mcc','Pool_816_AD006_indexed_R1_mcc','Pool_1202_AD008_indexed_R1_mcc','Pool_922_AD006_indexed_R1_mcc','Pool_993_AD002_indexed_R1_mcc','nuclei_508_mcc','Pool_572_AD008_indexed_R1_mcc','Pool_1206_AD010_indexed_R1_mcc','Pool_1216_AD008_indexed_R1_mcc','Pool_1259_AD010_indexed_R1_mcc','Pool_517_AD008_indexed_R1_mcc','Pool_952_AD002_indexed_R1_mcc','Pool_1226_AD008_indexed_R1_mcc','Pool_1114_AD006_indexed_R1_mcc','Pool_823_AD002_indexed_R1_mcc','Pool_999_AD002_indexed_R1_mcc','Pool_621_AD006_indexed_R1_mcc','Pool_697_AD010_indexed_R1_mcc','Pool_1234_AD008_indexed_R1_mcc','Pool_805_AD006_indexed_R1_mcc','Pool_6_AD002_indexed_R1_mcc','Pool_953_AD006_indexed_R1_mcc','Pool_1177_AD002_indexed_R1_mcc','Pool_844_AD010_indexed_R1_mcc','Pool_42_AD006_indexed_R1_mcc','Pool_1250_AD010_indexed_R1_mcc','Pool_858_AD008_indexed_R1_mcc','Pool_177_AD006_indexed_R1_mcc','Pool_752_AD008_indexed_R1_mcc','Pool_1217_AD008_indexed_R1_mcc','Pool_598_AD006_indexed_R1_mcc','Pool_653_AD002_indexed_R1_mcc','Pool_748_AD008_indexed_R1_mcc','Pool_16_AD002_indexed_R1_mcc','Pool_994_AD002_indexed_R1_mcc','Pool_360_AD008_indexed_R1_mcc','nuclei_336_mcc','Pool_560_AD008_indexed_R1_mcc','Pool_707_AD008_indexed_R1_mcc','nuclei_590_mcc','Pool_1013_AD008_indexed_R1_mcc','Pool_1274_AD002_indexed_R1_mcc','Pool_1104_AD010_indexed_R1_mcc','Pool_756_AD008_indexed_R1_mcc','Pool_1004_AD002_indexed_R1_mcc','Pool_895_AD010_indexed_R1_mcc','nuclei_588_mcc','Pool_1213_AD008_indexed_R1_mcc','nuclei_324_mcc','Pool_1242_AD010_indexed_R1_mcc','Pool_968_AD002_indexed_R1_mcc','Pool_922_AD002_indexed_R1_mcc','Pool_198_AD006_indexed_R1_mcc','Pool_1212_AD008_indexed_R1_mcc','nuclei_285_mcc','Pool_736_AD008_indexed_R1_mcc','Pool_1218_AD008_indexed_R1_mcc','Pool_654_AD006_indexed_R1_mcc','nuclei_566_mcc','Pool_1002_AD002_indexed_R1_mcc','Pool_1125_AD002_indexed_R1_mcc','Pool_632_AD002_indexed_R1_mcc','Pool_1031_AD008_indexed_R1_mcc','nuclei_574_mcc','Pool_1254_AD010_indexed_R1_mcc','Pool_1190_AD006_indexed_R1_mcc','Pool_1291_AD008_indexed_R1_mcc','nuclei_263_mcc','nuclei_337_mcc','Pool_1130_AD010_indexed_R1_mcc','Pool_309_AD010_indexed_R1_mcc','Pool_570_AD008_indexed_R1_mcc','Pool_460_AD002_indexed_R1_mcc','Pool_1216_AD006_indexed_R1_mcc','Pool_875_AD010_indexed_R1_mcc','Pool_1140_AD010_indexed_R1_mcc','Pool_171_AD002_indexed_R1_mcc','Pool_75_AD006_indexed_R1_mcc','Pool_1292_AD002_indexed_R1_mcc','nuclei_353_mcc','Pool_127_AD006_indexed_R1_mcc','Pool_124_AD002_indexed_R1_mcc','Pool_572_AD010_indexed_R1_mcc','Pool_136_AD006_indexed_R1_mcc','Pool_565_AD008_indexed_R1_mcc','Pool_1170_AD010_indexed_R1_mcc','nuclei_364_mcc','nuclei_544_mcc','Pool_304_AD008_indexed_R1_mcc','Pool_1_AD006_indexed_R1_mcc','Pool_497_AD010_indexed_R1_mcc','nuclei_429_mcc','Pool_433_AD006_indexed_R1_mcc','nuclei_301_mcc','Pool_1129_AD008_indexed_R1_mcc','Pool_477_AD002_indexed_R1_mcc','Pool_986_AD006_indexed_R1_mcc','Pool_1140_AD006_indexed_R1_mcc','nuclei_577_mcc','Pool_134_AD002_indexed_R1_mcc','Pool_102_AD002_indexed_R1_mcc','Pool_537_AD010_indexed_R1_mcc','Pool_875_AD008_indexed_R1_mcc','Pool_132_AD006_indexed_R1_mcc','Pool_1000_AD002_indexed_R1_mcc','Pool_1169_AD002_indexed_R1_mcc','Pool_87_AD006_indexed_R1_mcc','Pool_65_AD006_indexed_R1_mcc','Pool_560_AD010_indexed_R1_mcc','Pool_645_AD006_indexed_R1_mcc','Pool_200_AD006_indexed_R1_mcc','Pool_414_AD006_indexed_R1_mcc','Pool_97_AD002_indexed_R1_mcc','Pool_55_AD006_indexed_R1_mcc','Pool_921_AD006_indexed_R1_mcc','Pool_225_AD006_indexed_R1_mcc','Pool_368_AD008_indexed_R1_mcc','Pool_130_AD006_indexed_R1_mcc','Pool_659_AD006_indexed_R1_mcc','Pool_32_AD006_indexed_R1_mcc','Pool_1257_AD006_indexed_R1_mcc','Pool_1126_AD008_indexed_R1_mcc','Pool_439_AD002_indexed_R1_mcc','nuclei_597_mcc','Pool_63_AD002_indexed_R1_mcc','nuclei_290_mcc','Pool_1259_AD002_indexed_R1_mcc','Pool_928_AD002_indexed_R1_mcc','Pool_664_AD006_indexed_R1_mcc','Pool_301_AD008_indexed_R1_mcc','nuclei_567_mcc','Pool_1066_AD010_indexed_R1_mcc','Pool_1144_AD010_indexed_R1_mcc','Pool_1107_AD010_indexed_R1_mcc','nuclei_575_mcc','Pool_903_AD010_indexed_R1_mcc','Pool_1151_AD010_indexed_R1_mcc','Pool_1212_AD006_indexed_R1_mcc','Pool_1164_AD002_indexed_R1_mcc','Pool_9_AD002_indexed_R1_mcc','Pool_1240_AD002_indexed_R1_mcc','Pool_960_AD002_indexed_R1_mcc','Pool_181_AD002_indexed_R1_mcc','Pool_930_AD002_indexed_R1_mcc','Pool_507_AD008_indexed_R1_mcc','Pool_1222_AD002_indexed_R1_mcc','nuclei_321_mcc','Pool_902_AD010_indexed_R1_mcc','Pool_500_AD010_indexed_R1_mcc','Pool_140_AD002_indexed_R1_mcc','Pool_1128_AD002_indexed_R1_mcc','Pool_1115_AD002_indexed_R1_mcc','Pool_550_AD010_indexed_R1_mcc','Pool_765_AD010_indexed_R1_mcc','nuclei_550_mcc','Pool_306_AD010_indexed_R1_mcc','Pool_981_AD002_indexed_R1_mcc','Pool_1251_AD002_indexed_R1_mcc','Pool_278_AD006_indexed_R1_mcc','Pool_1068_AD008_indexed_R1_mcc','Pool_714_AD010_indexed_R1_mcc','nuclei_445_mcc','Pool_187_AD006_indexed_R1_mcc','Pool_505_AD010_indexed_R1_mcc','Pool_725_AD010_indexed_R1_mcc','Pool_1190_AD008_indexed_R1_mcc','Pool_409_AD006_indexed_R1_mcc','Pool_864_AD010_indexed_R1_mcc','Pool_521_AD010_indexed_R1_mcc','Pool_1107_AD002_indexed_R1_mcc','Pool_1142_AD008_indexed_R1_mcc','Pool_503_AD008_indexed_R1_mcc','Pool_1116_AD008_indexed_R1_mcc','Pool_1174_AD008_indexed_R1_mcc','Pool_164_AD002_indexed_R1_mcc','Pool_184_AD006_indexed_R1_mcc','Pool_1296_AD006_indexed_R1_mcc','Pool_1153_AD006_indexed_R1_mcc','nuclei_515_mcc','Pool_752_AD010_indexed_R1_mcc','Pool_491_AD008_indexed_R1_mcc','Pool_716_AD010_indexed_R1_mcc','nuclei_573_mcc','Pool_458_AD002_indexed_R1_mcc','Pool_1205_AD008_indexed_R1_mcc','Pool_1239_AD006_indexed_R1_mcc','Pool_1246_AD006_indexed_R1_mcc','Pool_358_AD010_indexed_R1_mcc','Pool_741_AD008_indexed_R1_mcc','nuclei_600_mcc','Pool_704_AD008_indexed_R1_mcc','Pool_117_AD006_indexed_R1_mcc','Pool_326_AD010_indexed_R1_mcc','Pool_1085_AD010_indexed_R1_mcc','Pool_1182_AD006_indexed_R1_mcc','mm_NeuN_pos_male_7wk_mcc','Pool_410_AD006_indexed_R1_mcc','nuclei_326_mcc','Pool_654_AD002_indexed_R1_mcc','Pool_1031_AD010_indexed_R1_mcc','Pool_1154_AD006_indexed_R1_mcc','Pool_1223_AD008_indexed_R1_mcc','Pool_628_AD002_indexed_R1_mcc','Pool_166_AD002_indexed_R1_mcc','Pool_1251_AD008_indexed_R1_mcc','Pool_907_AD008_indexed_R1_mcc','Pool_504_AD010_indexed_R1_mcc','Pool_533_AD010_indexed_R1_mcc','nuclei_571_mcc','Pool_1149_AD006_indexed_R1_mcc','Pool_897_AD010_indexed_R1_mcc','Pool_863_AD010_indexed_R1_mcc','Pool_1252_AD010_indexed_R1_mcc','Pool_1144_AD002_indexed_R1_mcc','Pool_859_AD010_indexed_R1_mcc','Pool_1199_AD002_indexed_R1_mcc','Pool_329_AD008_indexed_R1_mcc','Pool_1190_AD010_indexed_R1_mcc','Pool_948_AD006_indexed_R1_mcc','nuclei_268_mcc','Pool_179_AD002_indexed_R1_mcc','Pool_118_AD002_indexed_R1_mcc','Pool_804_AD006_indexed_R1_mcc','Pool_173_AD002_indexed_R1_mcc','Pool_310_AD008_indexed_R1_mcc','nuclei_332_mcc','Pool_671_AD006_indexed_R1_mcc','Pool_482_AD008_indexed_R1_mcc','Pool_118_AD006_indexed_R1_mcc','Pool_735_AD010_indexed_R1_mcc','Pool_831_AD002_indexed_R1_mcc','Pool_854_AD010_indexed_R1_mcc','Pool_509_AD010_indexed_R1_mcc','Pool_1220_AD008_indexed_R1_mcc','Pool_154_AD002_indexed_R1_mcc','Pool_139_AD002_indexed_R1_mcc','Pool_1266_AD008_indexed_R1_mcc','Pool_866_AD008_indexed_R1_mcc','Pool_456_AD002_indexed_R1_mcc','Pool_1089_AD010_indexed_R1_mcc','Pool_876_AD010_indexed_R1_mcc','Pool_993_AD006_indexed_R1_mcc','Pool_1293_AD008_indexed_R1_mcc','Pool_1183_AD006_indexed_R1_mcc','Pool_814_AD006_indexed_R1_mcc','Pool_1139_AD006_indexed_R1_mcc','Pool_582_AD006_indexed_R1_mcc','nuclei_309_mcc','Pool_56_AD002_indexed_R1_mcc','nuclei_579_mcc','Pool_1173_AD006_indexed_R1_mcc','Pool_130_AD002_indexed_R1_mcc','Pool_1168_AD006_indexed_R1_mcc','Pool_1120_AD002_indexed_R1_mcc','Pool_1250_AD008_indexed_R1_mcc','Pool_631_AD006_indexed_R1_mcc','Pool_1092_AD008_indexed_R1_mcc','Pool_1189_AD008_indexed_R1_mcc','Pool_755_AD010_indexed_R1_mcc','Pool_950_AD002_indexed_R1_mcc','Pool_857_AD010_indexed_R1_mcc','Pool_1069_AD008_indexed_R1_mcc','Pool_900_AD008_indexed_R1_mcc','Pool_1191_AD006_indexed_R1_mcc','Pool_768_AD008_indexed_R1_mcc','nuclei_427_mcc','Pool_649_AD006_indexed_R1_mcc','Pool_1264_AD010_indexed_R1_mcc','Pool_732_AD010_indexed_R1_mcc','Pool_935_AD002_indexed_R1_mcc','Pool_632_AD006_indexed_R1_mcc','Pool_1039_AD010_indexed_R1_mcc','Pool_932_AD002_indexed_R1_mcc','Pool_1153_AD002_indexed_R1_mcc','Pool_681_AD010_indexed_R1_mcc','Pool_470_AD002_indexed_R1_mcc','Pool_974_AD006_indexed_R1_mcc','Pool_956_AD002_indexed_R1_mcc','Pool_1287_AD008_indexed_R1_mcc','Pool_1255_AD002_indexed_R1_mcc','Pool_1289_AD010_indexed_R1_mcc','Pool_917_AD006_indexed_R1_mcc','Pool_955_AD006_indexed_R1_mcc','Pool_779_AD006_indexed_R1_mcc','Pool_629_AD006_indexed_R1_mcc','Pool_984_AD002_indexed_R1_mcc','Pool_1257_AD008_indexed_R1_mcc','Pool_704_AD010_indexed_R1_mcc','Pool_1001_AD006_indexed_R1_mcc','Pool_716_AD008_indexed_R1_mcc','Pool_673_AD008_indexed_R1_mcc','Pool_773_AD006_indexed_R1_mcc','Pool_1017_AD008_indexed_R1_mcc','Pool_978_AD002_indexed_R1_mcc','Pool_591_AD002_indexed_R1_mcc','Pool_1155_AD002_indexed_R1_mcc','Pool_1032_AD010_indexed_R1_mcc','Pool_1210_AD010_indexed_R1_mcc','Pool_1265_AD010_indexed_R1_mcc','Pool_1245_AD010_indexed_R1_mcc','Pool_1055_AD008_indexed_R1_mcc','Pool_1045_AD008_indexed_R1_mcc','nuclei_543_mcc','Pool_1032_AD008_indexed_R1_mcc','Pool_933_AD002_indexed_R1_mcc','Pool_637_AD006_indexed_R1_mcc','Pool_1143_AD002_indexed_R1_mcc','Pool_1192_AD008_indexed_R1_mcc','Pool_538_AD008_indexed_R1_mcc','Pool_63_AD006_indexed_R1_mcc','Pool_13_AD002_indexed_R1_mcc','Pool_115_AD006_indexed_R1_mcc','Pool_1168_AD008_indexed_R1_mcc','Pool_1167_AD008_indexed_R1_mcc','Pool_1194_AD010_indexed_R1_mcc','nuclei_277_mcc','Pool_473_AD006_indexed_R1_mcc','Pool_168_AD002_indexed_R1_mcc','nuclei_582_mcc','Pool_66_AD006_indexed_R1_mcc','Pool_1224_AD006_indexed_R1_mcc','Pool_270_AD006_indexed_R1_mcc','Pool_1215_AD002_indexed_R1_mcc','nuclei_335_mcc','Pool_1261_AD006_indexed_R1_mcc','Pool_168_AD006_indexed_R1_mcc','Pool_392_AD002_indexed_R1_mcc','Pool_422_AD002_indexed_R1_mcc','Pool_430_AD002_indexed_R1_mcc','Pool_17_AD006_indexed_R1_mcc','Pool_252_AD006_indexed_R1_mcc','nuclei_425_mcc','Pool_1270_AD006_indexed_R1_mcc','Pool_467_AD002_indexed_R1_mcc','Pool_406_AD002_indexed_R1_mcc','Pool_115_AD002_indexed_R1_mcc','Pool_107_AD006_indexed_R1_mcc','Pool_1254_AD002_indexed_R1_mcc','Pool_1156_AD010_indexed_R1_mcc','Pool_1176_AD008_indexed_R1_mcc','Pool_486_AD010_indexed_R1_mcc','Pool_40_AD002_indexed_R1_mcc','Pool_122_AD006_indexed_R1_mcc','Pool_161_AD002_indexed_R1_mcc','Pool_1245_AD002_indexed_R1_mcc','Pool_1177_AD008_indexed_R1_mcc','Pool_243_AD006_indexed_R1_mcc','Pool_75_AD002_indexed_R1_mcc','Pool_100_AD002_indexed_R1_mcc','Pool_1217_AD006_indexed_R1_mcc','nuclei_547_mcc','Pool_431_AD006_indexed_R1_mcc','Pool_472_AD002_indexed_R1_mcc','Pool_111_AD002_indexed_R1_mcc','Pool_144_AD002_indexed_R1_mcc','Pool_111_AD006_indexed_R1_mcc','Pool_1272_AD006_indexed_R1_mcc','nuclei_520_mcc','Pool_423_AD006_indexed_R1_mcc','Pool_403_AD002_indexed_R1_mcc','Pool_1113_AD008_indexed_R1_mcc','Pool_27_AD006_indexed_R1_mcc','Pool_294_AD010_indexed_R1_mcc','Pool_1293_AD006_indexed_R1_mcc','Pool_1295_AD002_indexed_R1_mcc','Pool_524_AD008_indexed_R1_mcc','Pool_507_AD010_indexed_R1_mcc','Pool_99_AD006_indexed_R1_mcc','Pool_33_AD006_indexed_R1_mcc','Pool_1194_AD008_indexed_R1_mcc','Pool_393_AD006_indexed_R1_mcc','Pool_157_AD002_indexed_R1_mcc','Pool_1261_AD002_indexed_R1_mcc','Pool_1157_AD010_indexed_R1_mcc','Pool_246_AD002_indexed_R1_mcc','Pool_485_AD008_indexed_R1_mcc','Pool_1189_AD010_indexed_R1_mcc','Pool_1153_AD008_indexed_R1_mcc','Pool_570_AD010_indexed_R1_mcc','Pool_169_AD002_indexed_R1_mcc','Pool_520_AD008_indexed_R1_mcc','Pool_47_AD002_indexed_R1_mcc','Pool_1286_AD002_indexed_R1_mcc','Pool_279_AD006_indexed_R1_mcc','Pool_536_AD008_indexed_R1_mcc','Pool_1204_AD006_indexed_R1_mcc','Pool_240_AD006_indexed_R1_mcc','Pool_167_AD006_indexed_R1_mcc','Pool_1127_AD010_indexed_R1_mcc','Pool_1289_AD006_indexed_R1_mcc','Pool_453_AD002_indexed_R1_mcc','Pool_80_AD002_indexed_R1_mcc','Pool_1271_AD002_indexed_R1_mcc','Pool_124_AD006_indexed_R1_mcc','nuclei_536_mcc','nuclei_356_mcc','Pool_89_AD006_indexed_R1_mcc','nuclei_265_mcc','Pool_1111_AD008_indexed_R1_mcc','Pool_1220_AD002_indexed_R1_mcc','nuclei_545_mcc','Pool_48_AD006_indexed_R1_mcc','Pool_545_AD010_indexed_R1_mcc','Pool_5_AD006_indexed_R1_mcc','Pool_244_AD006_indexed_R1_mcc','Pool_230_AD006_indexed_R1_mcc','Pool_183_AD006_indexed_R1_mcc','Pool_1196_AD008_indexed_R1_mcc','Pool_544_AD010_indexed_R1_mcc','Pool_287_AD006_indexed_R1_mcc','Pool_56_AD006_indexed_R1_mcc','Pool_95_AD006_indexed_R1_mcc','Pool_90_AD006_indexed_R1_mcc','Pool_576_AD008_indexed_R1_mcc','nuclei_509_mcc','Pool_471_AD002_indexed_R1_mcc','Pool_14_AD002_indexed_R1_mcc','nuclei_318_mcc','nuclei_542_mcc','AM_E1_mcc','Pool_1117_AD008_indexed_R1_mcc','Pool_146_AD002_indexed_R1_mcc','nuclei_331_mcc','Pool_1154_AD008_indexed_R1_mcc','Pool_1113_AD010_indexed_R1_mcc','Pool_510_AD010_indexed_R1_mcc','nuclei_589_mcc','Pool_82_AD006_indexed_R1_mcc','Pool_1283_AD002_indexed_R1_mcc','Pool_31_AD002_indexed_R1_mcc','Pool_19_AD006_indexed_R1_mcc','Pool_25_AD006_indexed_R1_mcc','Pool_254_AD006_indexed_R1_mcc','Pool_1167_AD010_indexed_R1_mcc','Pool_1230_AD006_indexed_R1_mcc','Pool_1213_AD002_indexed_R1_mcc','Pool_341_AD010_indexed_R1_mcc','Pool_83_AD006_indexed_R1_mcc','Pool_462_AD006_indexed_R1_mcc','Pool_1154_AD010_indexed_R1_mcc','Pool_556_AD008_indexed_R1_mcc','Pool_193_AD006_indexed_R1_mcc','nuclei_592_mcc','Pool_1236_AD002_indexed_R1_mcc','Pool_126_AD002_indexed_R1_mcc','Pool_73_AD006_indexed_R1_mcc','Pool_536_AD010_indexed_R1_mcc','Pool_1277_AD006_indexed_R1_mcc','Pool_492_AD010_indexed_R1_mcc','Pool_460_AD006_indexed_R1_mcc','Pool_133_AD002_indexed_R1_mcc','Pool_446_AD002_indexed_R1_mcc','Pool_519_AD010_indexed_R1_mcc','Pool_481_AD008_indexed_R1_mcc','Pool_1287_AD006_indexed_R1_mcc','Pool_1262_AD002_indexed_R1_mcc','Pool_1112_AD008_indexed_R1_mcc','Pool_526_AD008_indexed_R1_mcc','Pool_336_AD010_indexed_R1_mcc','Pool_26_AD006_indexed_R1_mcc','Pool_248_AD006_indexed_R1_mcc','Pool_106_AD002_indexed_R1_mcc','Pool_1262_AD006_indexed_R1_mcc','Pool_1178_AD008_indexed_R1_mcc','Pool_164_AD006_indexed_R1_mcc','Pool_1137_AD010_indexed_R1_mcc','Pool_1151_AD008_indexed_R1_mcc','Pool_463_AD006_indexed_R1_mcc','Pool_468_AD002_indexed_R1_mcc','Pool_464_AD006_indexed_R1_mcc','Pool_3_AD006_indexed_R1_mcc','nuclei_342_mcc','Pool_61_AD002_indexed_R1_mcc','Pool_165_AD006_indexed_R1_mcc','Pool_1143_AD010_indexed_R1_mcc','Pool_1172_AD008_indexed_R1_mcc','Pool_1157_AD008_indexed_R1_mcc','Pool_466_AD006_indexed_R1_mcc','Pool_829_AD006_indexed_R1_mcc','Pool_1065_AD008_indexed_R1_mcc','nuclei_533_mcc','Pool_1202_AD006_indexed_R1_mcc','nuclei_358_mcc','Pool_1124_AD006_indexed_R1_mcc','Pool_80_AD006_indexed_R1_mcc','Pool_422_AD006_indexed_R1_mcc','Pool_224_AD002_indexed_R1_mcc','Pool_107_AD002_indexed_R1_mcc','Pool_479_AD006_indexed_R1_mcc','Pool_238_AD006_indexed_R1_mcc','nuclei_325_mcc','Pool_26_AD002_indexed_R1_mcc','Pool_1178_AD006_indexed_R1_mcc','Pool_121_AD006_indexed_R1_mcc','Pool_862_AD008_indexed_R1_mcc','Pool_1254_AD006_indexed_R1_mcc','Pool_182_AD006_indexed_R1_mcc','Pool_1180_AD008_indexed_R1_mcc','Pool_848_AD010_indexed_R1_mcc','Pool_832_AD002_indexed_R1_mcc','Pool_685_AD008_indexed_R1_mcc','nuclei_537_mcc','Pool_1177_AD010_indexed_R1_mcc','Pool_633_AD002_indexed_R1_mcc','Pool_936_AD002_indexed_R1_mcc','Pool_1225_AD008_indexed_R1_mcc','nuclei_283_mcc','Pool_1049_AD010_indexed_R1_mcc','Pool_895_AD008_indexed_R1_mcc','Pool_125_AD006_indexed_R1_mcc','Pool_671_AD002_indexed_R1_mcc','nuclei_581_mcc','Pool_1054_AD010_indexed_R1_mcc','Pool_322_AD010_indexed_R1_mcc','Pool_923_AD006_indexed_R1_mcc','Pool_488_AD008_indexed_R1_mcc','Pool_159_AD006_indexed_R1_mcc','Pool_558_AD010_indexed_R1_mcc','Pool_113_AD002_indexed_R1_mcc','Pool_1010_AD008_indexed_R1_mcc','Pool_918_AD002_indexed_R1_mcc','Pool_1265_AD008_indexed_R1_mcc','nuclei_316_mcc','nuclei_319_mcc','Pool_981_AD006_indexed_R1_mcc','Pool_1275_AD002_indexed_R1_mcc','Pool_568_AD008_indexed_R1_mcc','Pool_131_AD002_indexed_R1_mcc','Pool_564_AD010_indexed_R1_mcc','Pool_512_AD010_indexed_R1_mcc','Pool_519_AD008_indexed_R1_mcc','Pool_35_AD006_indexed_R1_mcc','Pool_438_AD006_indexed_R1_mcc','Pool_1146_AD008_indexed_R1_mcc','Pool_194_AD006_indexed_R1_mcc','nuclei_264_mcc','Pool_501_AD010_indexed_R1_mcc','Pool_149_AD006_indexed_R1_mcc','Pool_1198_AD008_indexed_R1_mcc','Pool_1221_AD006_indexed_R1_mcc','Pool_1018_AD010_indexed_R1_mcc','Pool_1278_AD002_indexed_R1_mcc','nuclei_313_mcc','Pool_1133_AD008_indexed_R1_mcc','Pool_31_AD006_indexed_R1_mcc','Pool_452_AD002_indexed_R1_mcc','Pool_156_AD006_indexed_R1_mcc','Pool_280_AD006_indexed_R1_mcc','Pool_571_AD010_indexed_R1_mcc','nuclei_282_mcc','Pool_37_AD006_indexed_R1_mcc','Pool_186_AD006_indexed_R1_mcc','Pool_347_AD010_indexed_R1_mcc','Pool_1282_AD002_indexed_R1_mcc','Pool_447_AD006_indexed_R1_mcc','Pool_426_AD006_indexed_R1_mcc','Pool_1192_AD010_indexed_R1_mcc','Pool_68_AD006_indexed_R1_mcc','Pool_1272_AD002_indexed_R1_mcc','Pool_171_AD006_indexed_R1_mcc','Pool_1278_AD006_indexed_R1_mcc','nuclei_354_mcc','Pool_444_AD006_indexed_R1_mcc','Pool_1183_AD010_indexed_R1_mcc','Pool_1279_AD002_indexed_R1_mcc','Pool_566_AD010_indexed_R1_mcc','Pool_1200_AD010_indexed_R1_mcc','Pool_77_AD006_indexed_R1_mcc','nuclei_538_mcc','Pool_916_AD002_indexed_R1_mcc','Pool_169_AD006_indexed_R1_mcc','Pool_146_AD006_indexed_R1_mcc','Pool_1138_AD008_indexed_R1_mcc','Pool_1176_AD010_indexed_R1_mcc','Pool_156_AD002_indexed_R1_mcc','nuclei_559_mcc','Pool_195_AD002_indexed_R1_mcc','nuclei_302_mcc','nuclei_516_mcc','Pool_819_AD006_indexed_R1_mcc','Pool_240_AD002_indexed_R1_mcc','nuclei_281_mcc','nuclei_338_mcc','Pool_1173_AD008_indexed_R1_mcc','Pool_727_AD010_indexed_R1_mcc','nuclei_344_mcc','Pool_1211_AD002_indexed_R1_mcc','Pool_398_AD006_indexed_R1_mcc','Pool_419_AD002_indexed_R1_mcc','Pool_1251_AD006_indexed_R1_mcc','Pool_1198_AD010_indexed_R1_mcc','Pool_955_AD002_indexed_R1_mcc','Pool_1084_AD010_indexed_R1_mcc','Pool_457_AD006_indexed_R1_mcc','Pool_516_AD010_indexed_R1_mcc','Pool_1224_AD002_indexed_R1_mcc','Pool_1095_AD008_indexed_R1_mcc','Pool_556_AD010_indexed_R1_mcc','Pool_1132_AD008_indexed_R1_mcc','Pool_436_AD002_indexed_R1_mcc','nuclei_317_mcc','Pool_1159_AD008_indexed_R1_mcc','Pool_88_AD006_indexed_R1_mcc','Pool_36_AD006_indexed_R1_mcc','Pool_108_AD006_indexed_R1_mcc','Pool_1288_AD006_indexed_R1_mcc','Pool_571_AD008_indexed_R1_mcc','Pool_1250_AD002_indexed_R1_mcc','Pool_372_AD008_indexed_R1_mcc','Pool_1206_AD002_indexed_R1_mcc','Pool_184_AD002_indexed_R1_mcc','Pool_565_AD010_indexed_R1_mcc','Pool_1175_AD010_indexed_R1_mcc','Pool_1197_AD010_indexed_R1_mcc','Pool_91_AD002_indexed_R1_mcc','Pool_394_AD002_indexed_R1_mcc','Pool_190_AD006_indexed_R1_mcc','Pool_1203_AD002_indexed_R1_mcc','Pool_388_AD006_indexed_R1_mcc','Pool_1163_AD010_indexed_R1_mcc','nuclei_322_mcc','Pool_94_AD002_indexed_R1_mcc','Pool_1243_AD006_indexed_R1_mcc','Pool_135_AD006_indexed_R1_mcc','Pool_250_AD006_indexed_R1_mcc','Pool_414_AD002_indexed_R1_mcc','Pool_134_AD006_indexed_R1_mcc','Pool_195_AD006_indexed_R1_mcc','Pool_90_AD002_indexed_R1_mcc','Pool_540_AD008_indexed_R1_mcc','Pool_413_AD002_indexed_R1_mcc','Pool_283_AD006_indexed_R1_mcc','Pool_1283_AD006_indexed_R1_mcc','Pool_76_AD006_indexed_R1_mcc','Pool_53_AD006_indexed_R1_mcc','Pool_1110_AD008_indexed_R1_mcc','Pool_528_AD010_indexed_R1_mcc','Pool_62_AD006_indexed_R1_mcc','Pool_1204_AD002_indexed_R1_mcc','Pool_1169_AD010_indexed_R1_mcc','Pool_1110_AD010_indexed_R1_mcc','nuclei_339_mcc','nuclei_505_mcc','Pool_135_AD002_indexed_R1_mcc','Pool_371_AD010_indexed_R1_mcc','Pool_1145_AD008_indexed_R1_mcc','Pool_242_AD006_indexed_R1_mcc','Pool_120_AD002_indexed_R1_mcc','Pool_554_AD010_indexed_R1_mcc','Pool_524_AD010_indexed_R1_mcc','Pool_1181_AD008_indexed_R1_mcc','Pool_516_AD008_indexed_R1_mcc','Pool_272_AD006_indexed_R1_mcc','Pool_531_AD010_indexed_R1_mcc','Pool_566_AD008_indexed_R1_mcc','Pool_69_AD002_indexed_R1_mcc','Pool_1225_AD006_indexed_R1_mcc','Pool_1221_AD002_indexed_R1_mcc','Pool_1173_AD010_indexed_R1_mcc','Pool_216_AD002_indexed_R1_mcc','Pool_354_AD008_indexed_R1_mcc','Pool_436_AD006_indexed_R1_mcc','Pool_401_AD006_indexed_R1_mcc','Pool_98_AD006_indexed_R1_mcc','nuclei_564_mcc','Pool_1232_AD002_indexed_R1_mcc','Pool_557_AD008_indexed_R1_mcc','Pool_92_AD006_indexed_R1_mcc','Pool_424_AD006_indexed_R1_mcc','Pool_1125_AD008_indexed_R1_mcc','Pool_469_AD002_indexed_R1_mcc','nuclei_269_mcc','Pool_249_AD006_indexed_R1_mcc','Pool_181_AD006_indexed_R1_mcc','Pool_1257_AD002_indexed_R1_mcc','Pool_323_AD010_indexed_R1_mcc','Pool_455_AD006_indexed_R1_mcc','Pool_1187_AD008_indexed_R1_mcc','Pool_538_AD010_indexed_R1_mcc','Pool_1192_AD002_indexed_R1_mcc','Pool_48_AD002_indexed_R1_mcc','Pool_951_AD002_indexed_R1_mcc','Pool_49_AD002_indexed_R1_mcc','Pool_1145_AD006_indexed_R1_mcc','Pool_246_AD006_indexed_R1_mcc','Pool_147_AD002_indexed_R1_mcc','Pool_128_AD006_indexed_R1_mcc','nuclei_548_mcc','Pool_1282_AD006_indexed_R1_mcc','nuclei_349_mcc','Pool_428_AD002_indexed_R1_mcc','Pool_153_AD002_indexed_R1_mcc','nuclei_340_mcc','Pool_1195_AD010_indexed_R1_mcc','Pool_316_AD010_indexed_R1_mcc','Pool_70_AD006_indexed_R1_mcc','Pool_1152_AD008_indexed_R1_mcc','Pool_494_AD008_indexed_R1_mcc','Pool_145_AD002_indexed_R1_mcc','Pool_100_AD006_indexed_R1_mcc','Pool_515_AD010_indexed_R1_mcc','Pool_1207_AD002_indexed_R1_mcc','Pool_840_AD006_indexed_R1_mcc','Pool_1166_AD010_indexed_R1_mcc','Pool_1245_AD006_indexed_R1_mcc','Pool_1191_AD008_indexed_R1_mcc','Pool_384_AD008_indexed_R1_mcc','Pool_1280_AD002_indexed_R1_mcc','Pool_529_AD008_indexed_R1_mcc','Pool_534_AD010_indexed_R1_mcc','Pool_1269_AD002_indexed_R1_mcc','nuclei_267_mcc','nuclei_539_mcc','Pool_1143_AD008_indexed_R1_mcc','Pool_400_AD002_indexed_R1_mcc','Pool_991_AD002_indexed_R1_mcc','Pool_179_AD006_indexed_R1_mcc','Pool_1292_AD006_indexed_R1_mcc','Pool_57_AD002_indexed_R1_mcc','Pool_1199_AD010_indexed_R1_mcc','nuclei_293_mcc','Pool_173_AD006_indexed_R1_mcc','Pool_44_AD002_indexed_R1_mcc','Pool_551_AD010_indexed_R1_mcc','Pool_1276_AD002_indexed_R1_mcc','nuclei_330_mcc','nuclei_512_mcc','Pool_987_AD006_indexed_R1_mcc','Pool_54_AD002_indexed_R1_mcc','Pool_325_AD010_indexed_R1_mcc','Pool_403_AD006_indexed_R1_mcc','Pool_1043_AD010_indexed_R1_mcc','Pool_233_AD002_indexed_R1_mcc','Pool_1286_AD006_indexed_R1_mcc','Pool_1129_AD010_indexed_R1_mcc','Pool_172_AD006_indexed_R1_mcc','Pool_1273_AD010_indexed_R1_mcc','Pool_1250_AD006_indexed_R1_mcc','nuclei_525_mcc','Pool_498_AD010_indexed_R1_mcc','Pool_178_AD006_indexed_R1_mcc','Pool_518_AD008_indexed_R1_mcc','Pool_1120_AD010_indexed_R1_mcc','nuclei_278_mcc','Pool_284_AD006_indexed_R1_mcc','Pool_226_AD006_indexed_R1_mcc','Pool_52_AD006_indexed_R1_mcc','Pool_462_AD002_indexed_R1_mcc','Pool_368_AD010_indexed_R1_mcc','Pool_1166_AD008_indexed_R1_mcc','Pool_1271_AD006_indexed_R1_mcc','Pool_1130_AD008_indexed_R1_mcc','Pool_1226_AD006_indexed_R1_mcc','Pool_1280_AD006_indexed_R1_mcc','Pool_1095_AD010_indexed_R1_mcc','Pool_425_AD006_indexed_R1_mcc','Pool_521_AD008_indexed_R1_mcc','Pool_442_AD002_indexed_R1_mcc','Pool_154_AD006_indexed_R1_mcc','Pool_276_AD006_indexed_R1_mcc','Pool_497_AD008_indexed_R1_mcc','Pool_1273_AD002_indexed_R1_mcc','Pool_1210_AD002_indexed_R1_mcc','Pool_1116_AD010_indexed_R1_mcc','Pool_459_AD002_indexed_R1_mcc','Pool_1158_AD010_indexed_R1_mcc','Pool_50_AD002_indexed_R1_mcc','Pool_300_AD010_indexed_R1_mcc','Pool_267_AD006_indexed_R1_mcc','Pool_448_AD006_indexed_R1_mcc','Pool_148_AD002_indexed_R1_mcc','Pool_1186_AD008_indexed_R1_mcc','Pool_13_AD006_indexed_R1_mcc','Pool_1139_AD010_indexed_R1_mcc','Pool_102_AD006_indexed_R1_mcc','Pool_542_AD010_indexed_R1_mcc','Pool_508_AD010_indexed_R1_mcc','Pool_397_AD002_indexed_R1_mcc','Pool_534_AD008_indexed_R1_mcc','Pool_86_AD002_indexed_R1_mcc','Pool_526_AD010_indexed_R1_mcc','Pool_145_AD006_indexed_R1_mcc','Pool_574_AD010_indexed_R1_mcc','Pool_1159_AD010_indexed_R1_mcc','Pool_498_AD008_indexed_R1_mcc','Pool_487_AD010_indexed_R1_mcc','Pool_1237_AD002_indexed_R1_mcc','Pool_1197_AD008_indexed_R1_mcc','Pool_424_AD002_indexed_R1_mcc','Pool_219_AD006_indexed_R1_mcc','Pool_552_AD008_indexed_R1_mcc','Pool_258_AD006_indexed_R1_mcc','Pool_312_AD010_indexed_R1_mcc','Pool_10_AD006_indexed_R1_mcc','Pool_429_AD006_indexed_R1_mcc','AM_E2_mcc','Pool_218_AD006_indexed_R1_mcc','Pool_488_AD010_indexed_R1_mcc','Pool_474_AD006_indexed_R1_mcc','Pool_241_AD002_indexed_R1_mcc','Pool_12_AD006_indexed_R1_mcc','Pool_532_AD008_indexed_R1_mcc','Pool_60_AD006_indexed_R1_mcc','Pool_1160_AD008_indexed_R1_mcc','Pool_178_AD002_indexed_R1_mcc','Pool_112_AD006_indexed_R1_mcc','Pool_527_AD010_indexed_R1_mcc','Pool_1229_AD006_indexed_R1_mcc','Pool_396_AD002_indexed_R1_mcc','Pool_192_AD006_indexed_R1_mcc','Pool_1112_AD010_indexed_R1_mcc','Pool_1187_AD010_indexed_R1_mcc','Pool_17_AD002_indexed_R1_mcc','Pool_873_AD008_indexed_R1_mcc','Pool_70_AD002_indexed_R1_mcc','Pool_1290_AD006_indexed_R1_mcc','Pool_562_AD008_indexed_R1_mcc','Pool_76_AD002_indexed_R1_mcc','Pool_58_AD002_indexed_R1_mcc','Pool_302_AD010_indexed_R1_mcc','Pool_448_AD002_indexed_R1_mcc','Pool_1161_AD010_indexed_R1_mcc','Pool_176_AD006_indexed_R1_mcc','Pool_21_AD006_indexed_R1_mcc','Pool_266_AD006_indexed_R1_mcc','Pool_1291_AD006_indexed_R1_mcc','Pool_43_AD006_indexed_R1_mcc','Pool_457_AD002_indexed_R1_mcc','Pool_1138_AD010_indexed_R1_mcc','Pool_1267_AD006_indexed_R1_mcc','Pool_389_AD002_indexed_R1_mcc','Pool_286_AD006_indexed_R1_mcc','Pool_1162_AD010_indexed_R1_mcc','Pool_299_AD008_indexed_R1_mcc','Pool_54_AD006_indexed_R1_mcc','Pool_192_AD002_indexed_R1_mcc','Pool_180_AD002_indexed_R1_mcc','Pool_407_AD006_indexed_R1_mcc','Pool_219_AD002_indexed_R1_mcc','Pool_563_AD010_indexed_R1_mcc','Pool_1123_AD010_indexed_R1_mcc','Pool_231_AD002_indexed_R1_mcc','Pool_72_AD006_indexed_R1_mcc','Pool_224_AD006_indexed_R1_mcc','Pool_152_AD006_indexed_R1_mcc','Pool_1107_AD008_indexed_R1_mcc','Pool_1142_AD010_indexed_R1_mcc','Pool_1114_AD010_indexed_R1_mcc','Pool_475_AD006_indexed_R1_mcc','Pool_307_AD010_indexed_R1_mcc','Pool_84_AD006_indexed_R1_mcc','Pool_38_AD002_indexed_R1_mcc','Pool_83_AD002_indexed_R1_mcc','Pool_1215_AD006_indexed_R1_mcc','Pool_142_AD006_indexed_R1_mcc','Pool_528_AD008_indexed_R1_mcc','Pool_44_AD006_indexed_R1_mcc','Pool_105_AD006_indexed_R1_mcc','Pool_1223_AD006_indexed_R1_mcc','Pool_1219_AD006_indexed_R1_mcc','Pool_435_AD006_indexed_R1_mcc','Pool_158_AD006_indexed_R1_mcc','Pool_549_AD008_indexed_R1_mcc','Pool_432_AD006_indexed_R1_mcc','nuclei_549_mcc','Pool_161_AD006_indexed_R1_mcc','Pool_1287_AD002_indexed_R1_mcc','Pool_125_AD002_indexed_R1_mcc','Pool_1256_AD002_indexed_R1_mcc','Pool_474_AD002_indexed_R1_mcc','Pool_427_AD006_indexed_R1_mcc','Pool_74_AD006_indexed_R1_mcc','Pool_328_AD010_indexed_R1_mcc','Pool_1227_AD006_indexed_R1_mcc','Pool_540_AD010_indexed_R1_mcc','Pool_201_AD002_indexed_R1_mcc','Pool_212_AD006_indexed_R1_mcc','Pool_147_AD006_indexed_R1_mcc','Pool_98_AD002_indexed_R1_mcc','Pool_133_AD006_indexed_R1_mcc','Pool_1171_AD008_indexed_R1_mcc','Pool_1246_AD002_indexed_R1_mcc','Pool_14_AD006_indexed_R1_mcc','Pool_1266_AD006_indexed_R1_mcc','Pool_395_AD006_indexed_R1_mcc','Pool_112_AD002_indexed_R1_mcc','Pool_239_AD002_indexed_R1_mcc','Pool_445_AD006_indexed_R1_mcc','Pool_304_AD010_indexed_R1_mcc','Pool_256_AD006_indexed_R1_mcc','Pool_338_AD010_indexed_R1_mcc','Pool_1141_AD010_indexed_R1_mcc','Pool_73_AD002_indexed_R1_mcc','Pool_548_AD008_indexed_R1_mcc','Pool_7_AD006_indexed_R1_mcc','Pool_24_AD006_indexed_R1_mcc','Pool_1193_AD008_indexed_R1_mcc','Pool_575_AD008_indexed_R1_mcc','Pool_62_AD002_indexed_R1_mcc','Pool_404_AD006_indexed_R1_mcc','Pool_45_AD002_indexed_R1_mcc','Pool_28_AD002_indexed_R1_mcc','Pool_421_AD006_indexed_R1_mcc','Pool_58_AD006_indexed_R1_mcc','Pool_1208_AD006_indexed_R1_mcc','Pool_1188_AD010_indexed_R1_mcc','Pool_289_AD010_indexed_R1_mcc','Pool_413_AD006_indexed_R1_mcc','Pool_574_AD008_indexed_R1_mcc','Pool_214_AD006_indexed_R1_mcc','Pool_273_AD002_indexed_R1_mcc','Pool_468_AD006_indexed_R1_mcc','Pool_1289_AD002_indexed_R1_mcc','Pool_1211_AD006_indexed_R1_mcc','Pool_476_AD006_indexed_R1_mcc','Pool_149_AD002_indexed_R1_mcc','Pool_543_AD008_indexed_R1_mcc','Pool_1209_AD006_indexed_R1_mcc','Pool_255_AD006_indexed_R1_mcc','Pool_1252_AD006_indexed_R1_mcc','Pool_398_AD002_indexed_R1_mcc','Pool_261_AD006_indexed_R1_mcc','Pool_96_AD006_indexed_R1_mcc','Pool_191_AD002_indexed_R1_mcc','Pool_127_AD002_indexed_R1_mcc','Pool_420_AD006_indexed_R1_mcc','Pool_549_AD010_indexed_R1_mcc','Pool_489_AD008_indexed_R1_mcc','Pool_481_AD010_indexed_R1_mcc','Pool_485_AD010_indexed_R1_mcc','Pool_139_AD006_indexed_R1_mcc','Pool_81_AD006_indexed_R1_mcc','Pool_1124_AD010_indexed_R1_mcc','Pool_390_AD002_indexed_R1_mcc','Pool_395_AD002_indexed_R1_mcc','Pool_155_AD006_indexed_R1_mcc','Pool_116_AD002_indexed_R1_mcc','Pool_20_AD006_indexed_R1_mcc','Pool_1199_AD008_indexed_R1_mcc','Pool_38_AD006_indexed_R1_mcc','Pool_480_AD002_indexed_R1_mcc','Pool_450_AD002_indexed_R1_mcc','Pool_342_AD010_indexed_R1_mcc','Pool_370_AD010_indexed_R1_mcc','Pool_174_AD006_indexed_R1_mcc','Pool_131_AD006_indexed_R1_mcc','Pool_152_AD002_indexed_R1_mcc','Pool_1139_AD008_indexed_R1_mcc','Pool_1188_AD008_indexed_R1_mcc','Pool_108_AD002_indexed_R1_mcc','Pool_1212_AD002_indexed_R1_mcc','Pool_21_AD002_indexed_R1_mcc','Pool_635_AD006_indexed_R1_mcc','Pool_434_AD006_indexed_R1_mcc','Pool_1260_AD006_indexed_R1_mcc','Pool_539_AD010_indexed_R1_mcc','Pool_214_AD002_indexed_R1_mcc','Pool_397_AD006_indexed_R1_mcc','Pool_137_AD006_indexed_R1_mcc','Pool_480_AD006_indexed_R1_mcc','Pool_1125_AD010_indexed_R1_mcc','Pool_1288_AD002_indexed_R1_mcc','Pool_41_AD006_indexed_R1_mcc','Pool_182_AD002_indexed_R1_mcc','Pool_140_AD006_indexed_R1_mcc','Pool_1121_AD010_indexed_R1_mcc','Pool_1109_AD008_indexed_R1_mcc','Pool_82_AD002_indexed_R1_mcc','Pool_1115_AD010_indexed_R1_mcc','Pool_151_AD006_indexed_R1_mcc','Pool_148_AD006_indexed_R1_mcc','Pool_69_AD006_indexed_R1_mcc','Pool_176_AD002_indexed_R1_mcc','Pool_541_AD008_indexed_R1_mcc','Pool_103_AD006_indexed_R1_mcc','Pool_1249_AD002_indexed_R1_mcc','Pool_91_AD006_indexed_R1_mcc','Pool_29_AD006_indexed_R1_mcc','Pool_191_AD006_indexed_R1_mcc','Pool_1184_AD008_indexed_R1_mcc','Pool_311_AD010_indexed_R1_mcc','Pool_308_AD010_indexed_R1_mcc','Pool_477_AD006_indexed_R1_mcc','Pool_216_AD006_indexed_R1_mcc','Pool_511_AD010_indexed_R1_mcc','Pool_473_AD002_indexed_R1_mcc','Pool_567_AD010_indexed_R1_mcc','Pool_232_AD006_indexed_R1_mcc','Pool_225_AD002_indexed_R1_mcc','Pool_557_AD010_indexed_R1_mcc','Pool_382_AD010_indexed_R1_mcc','Pool_97_AD006_indexed_R1_mcc','Pool_138_AD002_indexed_R1_mcc','Pool_559_AD010_indexed_R1_mcc','Pool_377_AD010_indexed_R1_mcc','Pool_374_AD008_indexed_R1_mcc','Pool_85_AD002_indexed_R1_mcc','Pool_174_AD002_indexed_R1_mcc','Pool_1158_AD008_indexed_R1_mcc','Pool_1231_AD006_indexed_R1_mcc','Pool_1122_AD008_indexed_R1_mcc','nuclei_261_mcc','Pool_291_AD010_indexed_R1_mcc','Pool_1119_AD010_indexed_R1_mcc','Pool_420_AD002_indexed_R1_mcc','Pool_489_AD010_indexed_R1_mcc','Pool_283_AD002_indexed_R1_mcc','Pool_1140_AD008_indexed_R1_mcc','Pool_95_AD002_indexed_R1_mcc','Pool_399_AD002_indexed_R1_mcc','Pool_1153_AD010_indexed_R1_mcc','Pool_1127_AD008_indexed_R1_mcc','Pool_209_AD002_indexed_R1_mcc','Pool_1213_AD006_indexed_R1_mcc','Pool_1270_AD002_indexed_R1_mcc','Pool_379_AD010_indexed_R1_mcc','Pool_185_AD006_indexed_R1_mcc','Pool_1200_AD008_indexed_R1_mcc','Pool_350_AD010_indexed_R1_mcc','Pool_375_AD010_indexed_R1_mcc','Pool_1169_AD008_indexed_R1_mcc','Pool_1117_AD010_indexed_R1_mcc','Pool_223_AD006_indexed_R1_mcc','Pool_24_AD002_indexed_R1_mcc','Pool_523_AD010_indexed_R1_mcc','Pool_1255_AD006_indexed_R1_mcc','Pool_55_AD002_indexed_R1_mcc','Pool_210_AD006_indexed_R1_mcc','Pool_114_AD002_indexed_R1_mcc','Pool_456_AD006_indexed_R1_mcc','Pool_1279_AD006_indexed_R1_mcc','Pool_170_AD006_indexed_R1_mcc','Pool_311_AD008_indexed_R1_mcc','Pool_518_AD010_indexed_R1_mcc','Pool_486_AD008_indexed_R1_mcc','Pool_359_AD010_indexed_R1_mcc','Pool_183_AD002_indexed_R1_mcc','Pool_167_AD002_indexed_R1_mcc','Pool_295_AD010_indexed_R1_mcc','Pool_470_AD006_indexed_R1_mcc','Pool_463_AD002_indexed_R1_mcc','Pool_1277_AD002_indexed_R1_mcc','Pool_443_AD006_indexed_R1_mcc','Pool_221_AD006_indexed_R1_mcc','Pool_79_AD006_indexed_R1_mcc','Pool_151_AD002_indexed_R1_mcc','Pool_67_AD002_indexed_R1_mcc','Pool_408_AD006_indexed_R1_mcc','Pool_369_AD008_indexed_R1_mcc','Pool_408_AD002_indexed_R1_mcc','Pool_495_AD010_indexed_R1_mcc','Pool_541_AD010_indexed_R1_mcc','Pool_241_AD006_indexed_R1_mcc','Pool_1118_AD010_indexed_R1_mcc','Pool_415_AD006_indexed_R1_mcc','Pool_512_AD008_indexed_R1_mcc','Pool_496_AD010_indexed_R1_mcc','Pool_738_AD010_indexed_R1_mcc','nuclei_350_mcc','Pool_1192_AD006_indexed_R1_mcc','Pool_57_AD006_indexed_R1_mcc','Pool_1145_AD010_indexed_R1_mcc','Pool_992_AD006_indexed_R1_mcc','Pool_754_AD008_indexed_R1_mcc','Pool_136_AD002_indexed_R1_mcc','Pool_418_AD006_indexed_R1_mcc','Pool_927_AD006_indexed_R1_mcc','Pool_940_AD006_indexed_R1_mcc','Pool_584_AD006_indexed_R1_mcc','Pool_121_AD002_indexed_R1_mcc','Pool_1163_AD008_indexed_R1_mcc','Pool_455_AD002_indexed_R1_mcc','Pool_74_AD002_indexed_R1_mcc','Pool_39_AD006_indexed_R1_mcc','Pool_285_AD006_indexed_R1_mcc','Pool_502_AD010_indexed_R1_mcc','Pool_32_AD002_indexed_R1_mcc','Pool_1207_AD006_indexed_R1_mcc','Pool_12_AD002_indexed_R1_mcc','nuclei_355_mcc','Pool_396_AD006_indexed_R1_mcc','Pool_18_AD006_indexed_R1_mcc','Pool_555_AD010_indexed_R1_mcc','Pool_1161_AD008_indexed_R1_mcc','Pool_269_AD006_indexed_R1_mcc','Pool_93_AD002_indexed_R1_mcc','Pool_1144_AD008_indexed_R1_mcc','Pool_123_AD006_indexed_R1_mcc','Pool_299_AD010_indexed_R1_mcc','Pool_1178_AD010_indexed_R1_mcc','Pool_1248_AD002_indexed_R1_mcc','Pool_30_AD006_indexed_R1_mcc','Pool_335_AD010_indexed_R1_mcc','Pool_816_AD002_indexed_R1_mcc','Pool_552_AD010_indexed_R1_mcc','Pool_1256_AD006_indexed_R1_mcc','Pool_198_AD002_indexed_R1_mcc','nuclei_421_mcc','Pool_391_AD006_indexed_R1_mcc','Pool_411_AD006_indexed_R1_mcc','Pool_193_AD002_indexed_R1_mcc','nuclei_517_mcc','Pool_288_AD006_indexed_R1_mcc','Pool_428_AD006_indexed_R1_mcc','Pool_1263_AD006_indexed_R1_mcc','Pool_142_AD002_indexed_R1_mcc','Pool_28_AD006_indexed_R1_mcc','Pool_520_AD010_indexed_R1_mcc','Pool_51_AD006_indexed_R1_mcc','Pool_423_AD002_indexed_R1_mcc','nuclei_333_mcc','nuclei_351_mcc','Pool_1205_AD006_indexed_R1_mcc','Pool_568_AD010_indexed_R1_mcc','Pool_101_AD002_indexed_R1_mcc','Pool_562_AD010_indexed_R1_mcc','Pool_1156_AD008_indexed_R1_mcc','Pool_92_AD002_indexed_R1_mcc','Pool_366_AD008_indexed_R1_mcc','Pool_1226_AD010_indexed_R1_mcc','Pool_160_AD006_indexed_R1_mcc','nuclei_381_mcc','Pool_459_AD006_indexed_R1_mcc','Pool_1232_AD008_indexed_R1_mcc','Pool_227_AD006_indexed_R1_mcc','Pool_49_AD006_indexed_R1_mcc','Pool_84_AD002_indexed_R1_mcc','Pool_1274_AD006_indexed_R1_mcc','Pool_1155_AD008_indexed_R1_mcc','Pool_1266_AD002_indexed_R1_mcc','nuclei_292_mcc','nuclei_569_mcc','Pool_1147_AD010_indexed_R1_mcc','Pool_1174_AD010_indexed_R1_mcc','nuclei_308_mcc','Pool_409_AD002_indexed_R1_mcc','Pool_550_AD008_indexed_R1_mcc','Pool_188_AD006_indexed_R1_mcc','Pool_1214_AD002_indexed_R1_mcc','Pool_419_AD006_indexed_R1_mcc','Pool_61_AD006_indexed_R1_mcc','Pool_40_AD006_indexed_R1_mcc','Pool_11_AD006_indexed_R1_mcc','Pool_1182_AD010_indexed_R1_mcc','Pool_110_AD006_indexed_R1_mcc','Pool_776_AD006_indexed_R1_mcc','Pool_469_AD006_indexed_R1_mcc','Pool_296_AD008_indexed_R1_mcc','Pool_123_AD002_indexed_R1_mcc','Pool_535_AD010_indexed_R1_mcc','Pool_105_AD002_indexed_R1_mcc','Pool_175_AD006_indexed_R1_mcc','Pool_251_AD006_indexed_R1_mcc','nuclei_284_mcc','Pool_1042_AD008_indexed_R1_mcc','Pool_1210_AD006_indexed_R1_mcc','Pool_514_AD010_indexed_R1_mcc','nuclei_555_mcc','Pool_259_AD006_indexed_R1_mcc','Pool_1247_AD006_indexed_R1_mcc','Pool_1155_AD010_indexed_R1_mcc','Pool_1231_AD002_indexed_R1_mcc','Pool_51_AD002_indexed_R1_mcc','Pool_446_AD006_indexed_R1_mcc','nuclei_551_mcc','Pool_1239_AD002_indexed_R1_mcc','Pool_495_AD008_indexed_R1_mcc','Pool_271_AD006_indexed_R1_mcc','Pool_1196_AD010_indexed_R1_mcc','Pool_104_AD002_indexed_R1_mcc','Pool_567_AD008_indexed_R1_mcc','Pool_1241_AD006_indexed_R1_mcc','Pool_119_AD002_indexed_R1_mcc','Pool_450_AD006_indexed_R1_mcc','nuclei_524_mcc','Pool_529_AD010_indexed_R1_mcc','Pool_257_AD006_indexed_R1_mcc','Pool_1258_AD006_indexed_R1_mcc','Pool_1114_AD008_indexed_R1_mcc','Pool_160_AD002_indexed_R1_mcc','Pool_163_AD006_indexed_R1_mcc','Pool_23_AD006_indexed_R1_mcc','Pool_451_AD006_indexed_R1_mcc','Pool_561_AD008_indexed_R1_mcc','nuclei_523_mcc','Pool_387_AD002_indexed_R1_mcc','Pool_103_AD002_indexed_R1_mcc','Pool_229_AD006_indexed_R1_mcc','Pool_1259_AD006_indexed_R1_mcc','nuclei_507_mcc','Pool_16_AD006_indexed_R1_mcc','Pool_465_AD006_indexed_R1_mcc','Pool_87_AD002_indexed_R1_mcc','Pool_189_AD002_indexed_R1_mcc','Pool_194_AD002_indexed_R1_mcc','Pool_245_AD006_indexed_R1_mcc','Pool_555_AD008_indexed_R1_mcc','Pool_312_AD008_indexed_R1_mcc','Pool_437_AD002_indexed_R1_mcc','Pool_64_AD006_indexed_R1_mcc','Pool_313_AD010_indexed_R1_mcc','Pool_503_AD010_indexed_R1_mcc','nuclei_312_mcc','Pool_410_AD002_indexed_R1_mcc','Pool_263_AD006_indexed_R1_mcc','Pool_220_AD006_indexed_R1_mcc','Pool_1223_AD002_indexed_R1_mcc','nuclei_440_mcc','Pool_493_AD010_indexed_R1_mcc','Pool_120_AD006_indexed_R1_mcc','Pool_346_AD008_indexed_R1_mcc','Pool_432_AD002_indexed_R1_mcc','Pool_390_AD006_indexed_R1_mcc','Pool_180_AD006_indexed_R1_mcc','Pool_1149_AD008_indexed_R1_mcc','Pool_1185_AD008_indexed_R1_mcc','Pool_1164_AD008_indexed_R1_mcc','Pool_144_AD006_indexed_R1_mcc','Pool_472_AD006_indexed_R1_mcc','Pool_415_AD002_indexed_R1_mcc','Pool_205_AD006_indexed_R1_mcc','Pool_1108_AD010_indexed_R1_mcc','Pool_119_AD006_indexed_R1_mcc','Pool_537_AD008_indexed_R1_mcc','nuclei_506_mcc'],['nuclei_279_mcc','nuclei_518_mcc','nuclei_438_mcc','Pool_430_AD006_indexed_R1_mcc','Pool_1188_AD002_indexed_R1_mcc','Pool_979_AD002_indexed_R1_mcc','Pool_1285_AD002_indexed_R1_mcc','Pool_1020_AD008_indexed_R1_mcc','Pool_45_AD006_indexed_R1_mcc','Pool_1152_AD010_indexed_R1_mcc','Pool_674_AD008_indexed_R1_mcc','nuclei_598_mcc','Pool_1284_AD010_indexed_R1_mcc','Pool_1165_AD010_indexed_R1_mcc','Pool_1085_AD008_indexed_R1_mcc','Pool_926_AD006_indexed_R1_mcc','Pool_696_AD008_indexed_R1_mcc','Pool_1101_AD010_indexed_R1_mcc','Pool_647_AD006_indexed_R1_mcc','Pool_274_AD002_indexed_R1_mcc','Pool_1208_AD008_indexed_R1_mcc','Pool_1288_AD010_indexed_R1_mcc','Pool_1081_AD008_indexed_R1_mcc','Pool_1089_AD008_indexed_R1_mcc','Pool_734_AD010_indexed_R1_mcc','Pool_652_AD006_indexed_R1_mcc','Pool_764_AD008_indexed_R1_mcc','Pool_734_AD008_indexed_R1_mcc','Pool_900_AD010_indexed_R1_mcc','Pool_1071_AD010_indexed_R1_mcc','Pool_445_AD002_indexed_R1_mcc','Pool_962_AD006_indexed_R1_mcc','Pool_1241_AD010_indexed_R1_mcc','Pool_1050_AD008_indexed_R1_mcc','Pool_2_AD006_indexed_R1_mcc','Pool_1091_AD008_indexed_R1_mcc','Pool_1213_AD010_indexed_R1_mcc','Pool_1264_AD006_indexed_R1_mcc','Pool_796_AD002_indexed_R1_mcc','Pool_695_AD008_indexed_R1_mcc','Pool_999_AD006_indexed_R1_mcc','Pool_1122_AD006_indexed_R1_mcc','Pool_630_AD006_indexed_R1_mcc','Pool_647_AD002_indexed_R1_mcc','nuclei_424_mcc','Pool_865_AD010_indexed_R1_mcc','Pool_1087_AD010_indexed_R1_mcc','Pool_1021_AD010_indexed_R1_mcc','Pool_583_AD006_indexed_R1_mcc','Pool_1002_AD006_indexed_R1_mcc','Pool_989_AD002_indexed_R1_mcc','Pool_913_AD002_indexed_R1_mcc','Pool_971_AD006_indexed_R1_mcc','Pool_925_AD002_indexed_R1_mcc','Pool_1197_AD002_indexed_R1_mcc','Pool_1202_AD010_indexed_R1_mcc','Pool_861_AD010_indexed_R1_mcc','Pool_1258_AD010_indexed_R1_mcc','Pool_1114_AD002_indexed_R1_mcc','Pool_759_AD008_indexed_R1_mcc','Pool_877_AD010_indexed_R1_mcc','Pool_689_AD010_indexed_R1_mcc','Pool_797_AD006_indexed_R1_mcc','Pool_687_AD008_indexed_R1_mcc','Pool_1080_AD008_indexed_R1_mcc','Pool_1094_AD008_indexed_R1_mcc','Pool_1017_AD010_indexed_R1_mcc','Pool_945_AD006_indexed_R1_mcc','Pool_608_AD006_indexed_R1_mcc','Pool_830_AD006_indexed_R1_mcc','Pool_997_AD002_indexed_R1_mcc','Pool_976_AD006_indexed_R1_mcc','Pool_830_AD002_indexed_R1_mcc','Pool_591_AD006_indexed_R1_mcc','Pool_746_AD010_indexed_R1_mcc','Pool_717_AD008_indexed_R1_mcc','Pool_740_AD008_indexed_R1_mcc','Pool_1086_AD010_indexed_R1_mcc','Pool_1211_AD008_indexed_R1_mcc','Pool_929_AD002_indexed_R1_mcc','Pool_1243_AD008_indexed_R1_mcc','Pool_954_AD002_indexed_R1_mcc','Pool_740_AD010_indexed_R1_mcc','Pool_866_AD010_indexed_R1_mcc','Pool_1065_AD010_indexed_R1_mcc','Pool_901_AD010_indexed_R1_mcc','Pool_673_AD010_indexed_R1_mcc','Pool_888_AD008_indexed_R1_mcc','Pool_971_AD002_indexed_R1_mcc','Pool_1180_AD010_indexed_R1_mcc','Pool_1280_AD010_indexed_R1_mcc','Pool_756_AD010_indexed_R1_mcc','Pool_946_AD006_indexed_R1_mcc','Pool_937_AD002_indexed_R1_mcc','Pool_983_AD006_indexed_R1_mcc','Pool_909_AD008_indexed_R1_mcc','Pool_924_AD006_indexed_R1_mcc','Pool_1207_AD008_indexed_R1_mcc','Pool_820_AD002_indexed_R1_mcc','Pool_667_AD006_indexed_R1_mcc','Pool_946_AD002_indexed_R1_mcc','Pool_700_AD008_indexed_R1_mcc','Pool_609_AD006_indexed_R1_mcc','Pool_588_AD002_indexed_R1_mcc','Pool_721_AD008_indexed_R1_mcc','Pool_853_AD008_indexed_R1_mcc','Pool_743_AD008_indexed_R1_mcc','Pool_945_AD002_indexed_R1_mcc','Pool_815_AD002_indexed_R1_mcc','Pool_835_AD002_indexed_R1_mcc','Pool_998_AD002_indexed_R1_mcc','Pool_1261_AD010_indexed_R1_mcc','Pool_960_AD006_indexed_R1_mcc','Pool_1272_AD008_indexed_R1_mcc','Pool_997_AD006_indexed_R1_mcc','Pool_881_AD008_indexed_R1_mcc','nuclei_347_mcc','Pool_881_AD010_indexed_R1_mcc','Pool_622_AD006_indexed_R1_mcc','Pool_1279_AD010_indexed_R1_mcc','Pool_1176_AD006_indexed_R1_mcc','Pool_763_AD008_indexed_R1_mcc','Pool_1209_AD010_indexed_R1_mcc','Pool_893_AD008_indexed_R1_mcc','Pool_1221_AD010_indexed_R1_mcc','Pool_834_AD002_indexed_R1_mcc','Pool_1247_AD010_indexed_R1_mcc','Pool_949_AD006_indexed_R1_mcc','Pool_855_AD008_indexed_R1_mcc','Pool_799_AD006_indexed_R1_mcc','Pool_641_AD006_indexed_R1_mcc','Pool_995_AD002_indexed_R1_mcc','Pool_894_AD010_indexed_R1_mcc','Pool_751_AD010_indexed_R1_mcc','Pool_746_AD008_indexed_R1_mcc','Pool_1272_AD010_indexed_R1_mcc','Pool_917_AD002_indexed_R1_mcc','Pool_941_AD006_indexed_R1_mcc','Pool_920_AD002_indexed_R1_mcc','Pool_728_AD010_indexed_R1_mcc','Pool_690_AD010_indexed_R1_mcc','Pool_1203_AD008_indexed_R1_mcc','nuclei_526_mcc','Pool_1047_AD010_indexed_R1_mcc','Pool_1259_AD008_indexed_R1_mcc','Pool_1246_AD010_indexed_R1_mcc','Pool_684_AD010_indexed_R1_mcc','nuclei_455_mcc','Pool_665_AD006_indexed_R1_mcc','nuclei_554_mcc','Pool_847_AD010_indexed_R1_mcc','Pool_1024_AD010_indexed_R1_mcc','Pool_625_AD006_indexed_R1_mcc','nuclei_576_mcc','Pool_637_AD002_indexed_R1_mcc','Pool_1230_AD010_indexed_R1_mcc','Pool_961_AD002_indexed_R1_mcc','Pool_766_AD010_indexed_R1_mcc','nuclei_572_mcc','Pool_1116_AD002_indexed_R1_mcc','nuclei_528_mcc','nuclei_578_mcc','Pool_581_AD006_indexed_R1_mcc','Pool_1269_AD008_indexed_R1_mcc','Pool_885_AD010_indexed_R1_mcc','Pool_1066_AD008_indexed_R1_mcc','Pool_1248_AD010_indexed_R1_mcc','Pool_1094_AD010_indexed_R1_mcc','Pool_645_AD002_indexed_R1_mcc','Pool_662_AD006_indexed_R1_mcc','Pool_1166_AD006_indexed_R1_mcc','Pool_822_AD002_indexed_R1_mcc','Pool_1290_AD010_indexed_R1_mcc','Pool_1075_AD008_indexed_R1_mcc','Pool_1256_AD010_indexed_R1_mcc','Pool_1196_AD006_indexed_R1_mcc','Pool_1201_AD010_indexed_R1_mcc','Pool_967_AD002_indexed_R1_mcc','Pool_726_AD008_indexed_R1_mcc','Pool_849_AD010_indexed_R1_mcc','Pool_1125_AD006_indexed_R1_mcc','Pool_1247_AD008_indexed_R1_mcc','Pool_874_AD010_indexed_R1_mcc','Pool_1121_AD006_indexed_R1_mcc','Pool_1028_AD008_indexed_R1_mcc','Pool_1117_AD002_indexed_R1_mcc','Pool_1073_AD008_indexed_R1_mcc','Pool_580_AD006_indexed_R1_mcc','Pool_1296_AD008_indexed_R1_mcc','Pool_817_AD006_indexed_R1_mcc','nuclei_270_mcc','Pool_672_AD002_indexed_R1_mcc','Pool_1054_AD008_indexed_R1_mcc','Pool_1255_AD010_indexed_R1_mcc','Pool_1064_AD010_indexed_R1_mcc','nuclei_510_mcc','Pool_1194_AD002_indexed_R1_mcc','Pool_902_AD008_indexed_R1_mcc','Pool_1222_AD010_indexed_R1_mcc','Pool_1171_AD002_indexed_R1_mcc','Pool_1245_AD008_indexed_R1_mcc','Pool_827_AD006_indexed_R1_mcc','Pool_672_AD006_indexed_R1_mcc','Pool_988_AD006_indexed_R1_mcc','Pool_1139_AD002_indexed_R1_mcc','Pool_733_AD008_indexed_R1_mcc','Pool_1224_AD008_indexed_R1_mcc','Pool_646_AD006_indexed_R1_mcc','Pool_855_AD010_indexed_R1_mcc','Pool_1190_AD002_indexed_R1_mcc','Pool_764_AD010_indexed_R1_mcc','Pool_698_AD010_indexed_R1_mcc','Pool_1006_AD002_indexed_R1_mcc','nuclei_580_mcc','Pool_868_AD008_indexed_R1_mcc','Pool_1281_AD010_indexed_R1_mcc','Pool_1290_AD008_indexed_R1_mcc','Pool_846_AD010_indexed_R1_mcc','nuclei_556_mcc','Pool_650_AD002_indexed_R1_mcc','Pool_844_AD008_indexed_R1_mcc','Pool_1025_AD008_indexed_R1_mcc','Pool_1185_AD002_indexed_R1_mcc','Pool_1142_AD002_indexed_R1_mcc','Pool_1274_AD010_indexed_R1_mcc','Pool_798_AD006_indexed_R1_mcc','Pool_1019_AD008_indexed_R1_mcc','Pool_643_AD002_indexed_R1_mcc','Pool_617_AD002_indexed_R1_mcc','Pool_972_AD002_indexed_R1_mcc','Pool_974_AD002_indexed_R1_mcc','Pool_1288_AD008_indexed_R1_mcc','Pool_1138_AD006_indexed_R1_mcc','Pool_1178_AD002_indexed_R1_mcc','Pool_1099_AD008_indexed_R1_mcc','Pool_1158_AD002_indexed_R1_mcc','Pool_913_AD006_indexed_R1_mcc','Pool_653_AD006_indexed_R1_mcc','Pool_910_AD008_indexed_R1_mcc','Pool_718_AD008_indexed_R1_mcc','Pool_882_AD010_indexed_R1_mcc','Pool_1011_AD008_indexed_R1_mcc','Pool_969_AD006_indexed_R1_mcc','Pool_980_AD002_indexed_R1_mcc','Pool_1048_AD010_indexed_R1_mcc','Pool_836_AD006_indexed_R1_mcc','Pool_711_AD008_indexed_R1_mcc','nuclei_275_mcc','nuclei_345_mcc','Pool_1121_AD002_indexed_R1_mcc','Pool_979_AD006_indexed_R1_mcc','Pool_1173_AD002_indexed_R1_mcc','Pool_951_AD006_indexed_R1_mcc','Pool_729_AD010_indexed_R1_mcc','Pool_1055_AD010_indexed_R1_mcc','Pool_670_AD002_indexed_R1_mcc','Pool_1271_AD008_indexed_R1_mcc','Pool_1233_AD010_indexed_R1_mcc','Pool_885_AD008_indexed_R1_mcc','Pool_669_AD006_indexed_R1_mcc','Pool_1161_AD002_indexed_R1_mcc','Pool_712_AD010_indexed_R1_mcc','Pool_876_AD008_indexed_R1_mcc','Pool_587_AD006_indexed_R1_mcc','Pool_821_AD002_indexed_R1_mcc','Pool_925_AD006_indexed_R1_mcc','Pool_1244_AD008_indexed_R1_mcc','Pool_1252_AD008_indexed_R1_mcc','Pool_1248_AD008_indexed_R1_mcc','Pool_1277_AD008_indexed_R1_mcc','Pool_952_AD006_indexed_R1_mcc','Pool_1170_AD006_indexed_R1_mcc','Pool_931_AD002_indexed_R1_mcc','Pool_839_AD002_indexed_R1_mcc','Pool_1282_AD010_indexed_R1_mcc','Pool_1059_AD008_indexed_R1_mcc','Pool_991_AD006_indexed_R1_mcc','Pool_852_AD008_indexed_R1_mcc','Pool_705_AD008_indexed_R1_mcc','Pool_648_AD006_indexed_R1_mcc','Pool_1294_AD008_indexed_R1_mcc','Pool_1083_AD010_indexed_R1_mcc','Pool_847_AD008_indexed_R1_mcc','Pool_1052_AD010_indexed_R1_mcc','Pool_870_AD010_indexed_R1_mcc','nuclei_343_mcc','Pool_860_AD010_indexed_R1_mcc','Pool_661_AD002_indexed_R1_mcc','Pool_1129_AD002_indexed_R1_mcc','Pool_1113_AD002_indexed_R1_mcc','Pool_1169_AD006_indexed_R1_mcc','Pool_1287_AD010_indexed_R1_mcc','Pool_985_AD006_indexed_R1_mcc','Pool_1023_AD010_indexed_R1_mcc','Pool_1018_AD008_indexed_R1_mcc','Pool_1294_AD010_indexed_R1_mcc','Pool_1239_AD008_indexed_R1_mcc','Pool_1238_AD008_indexed_R1_mcc','Pool_970_AD002_indexed_R1_mcc','nuclei_546_mcc','Pool_837_AD006_indexed_R1_mcc','Pool_784_AD006_indexed_R1_mcc','Pool_1199_AD006_indexed_R1_mcc','Pool_713_AD010_indexed_R1_mcc','Pool_1249_AD010_indexed_R1_mcc','Pool_1200_AD006_indexed_R1_mcc','Pool_1198_AD006_indexed_R1_mcc','Pool_928_AD006_indexed_R1_mcc','nuclei_560_mcc','Pool_1103_AD008_indexed_R1_mcc','Pool_1286_AD008_indexed_R1_mcc','Pool_1197_AD006_indexed_R1_mcc','Pool_1083_AD008_indexed_R1_mcc','Pool_1008_AD002_indexed_R1_mcc','Pool_878_AD010_indexed_R1_mcc','Pool_596_AD006_indexed_R1_mcc','Pool_1086_AD008_indexed_R1_mcc','Pool_871_AD010_indexed_R1_mcc','Pool_1182_AD002_indexed_R1_mcc','Pool_1157_AD006_indexed_R1_mcc','Pool_1041_AD010_indexed_R1_mcc','Pool_1078_AD008_indexed_R1_mcc','Pool_1264_AD008_indexed_R1_mcc','Pool_857_AD008_indexed_R1_mcc','Pool_1278_AD008_indexed_R1_mcc','Pool_1012_AD008_indexed_R1_mcc','Pool_730_AD008_indexed_R1_mcc','Pool_943_AD002_indexed_R1_mcc','Pool_599_AD006_indexed_R1_mcc','Pool_1270_AD010_indexed_R1_mcc','Pool_886_AD008_indexed_R1_mcc','nuclei_420_mcc','Pool_691_AD008_indexed_R1_mcc','Pool_799_AD002_indexed_R1_mcc','Pool_1285_AD010_indexed_R1_mcc','nuclei_558_mcc','Pool_863_AD008_indexed_R1_mcc','Pool_1076_AD010_indexed_R1_mcc','Pool_860_AD008_indexed_R1_mcc','Pool_651_AD006_indexed_R1_mcc','Pool_789_AD002_indexed_R1_mcc','Pool_1225_AD010_indexed_R1_mcc','Pool_742_AD008_indexed_R1_mcc','Pool_1263_AD010_indexed_R1_mcc','Pool_634_AD002_indexed_R1_mcc','Pool_737_AD010_indexed_R1_mcc','Pool_856_AD008_indexed_R1_mcc','Pool_1000_AD006_indexed_R1_mcc','Pool_1080_AD010_indexed_R1_mcc','Pool_1063_AD008_indexed_R1_mcc','Pool_691_AD010_indexed_R1_mcc','Pool_906_AD010_indexed_R1_mcc','Pool_726_AD010_indexed_R1_mcc','Pool_975_AD006_indexed_R1_mcc','Pool_1014_AD008_indexed_R1_mcc','Pool_769_AD006_indexed_R1_mcc','nuclei_314_mcc','Pool_648_AD002_indexed_R1_mcc','Pool_1181_AD006_indexed_R1_mcc','Pool_815_AD006_indexed_R1_mcc','Pool_803_AD002_indexed_R1_mcc','nuclei_535_mcc','Pool_710_AD010_indexed_R1_mcc','Pool_1231_AD010_indexed_R1_mcc','Pool_878_AD008_indexed_R1_mcc','Pool_939_AD006_indexed_R1_mcc','Pool_1072_AD010_indexed_R1_mcc','Pool_1025_AD010_indexed_R1_mcc','Pool_879_AD008_indexed_R1_mcc','Pool_1060_AD008_indexed_R1_mcc','Pool_1218_AD010_indexed_R1_mcc','Pool_840_AD002_indexed_R1_mcc','Pool_938_AD006_indexed_R1_mcc','Pool_795_AD006_indexed_R1_mcc','Pool_784_AD002_indexed_R1_mcc','Pool_814_AD002_indexed_R1_mcc','Pool_890_AD010_indexed_R1_mcc','Pool_1207_AD010_indexed_R1_mcc','Pool_1008_AD006_indexed_R1_mcc','Pool_938_AD002_indexed_R1_mcc','Pool_1235_AD008_indexed_R1_mcc','Pool_973_AD006_indexed_R1_mcc','Pool_741_AD010_indexed_R1_mcc','Pool_640_AD002_indexed_R1_mcc','Pool_800_AD006_indexed_R1_mcc','Pool_1295_AD008_indexed_R1_mcc','Pool_965_AD002_indexed_R1_mcc','Pool_901_AD008_indexed_R1_mcc','Pool_1118_AD002_indexed_R1_mcc','Pool_926_AD002_indexed_R1_mcc','Pool_1119_AD002_indexed_R1_mcc','Pool_1115_AD006_indexed_R1_mcc','Pool_1292_AD010_indexed_R1_mcc','Pool_1186_AD006_indexed_R1_mcc','Pool_1058_AD010_indexed_R1_mcc','Pool_963_AD006_indexed_R1_mcc','Pool_897_AD008_indexed_R1_mcc','Pool_1151_AD006_indexed_R1_mcc','Pool_1183_AD002_indexed_R1_mcc','Pool_1028_AD010_indexed_R1_mcc','Pool_626_AD006_indexed_R1_mcc','Pool_748_AD010_indexed_R1_mcc','Pool_1090_AD008_indexed_R1_mcc','Pool_577_AD006_indexed_R1_mcc','Pool_1276_AD010_indexed_R1_mcc','Pool_1180_AD006_indexed_R1_mcc','Pool_1130_AD002_indexed_R1_mcc','Pool_825_AD006_indexed_R1_mcc','Pool_1132_AD006_indexed_R1_mcc','Pool_1275_AD008_indexed_R1_mcc','Pool_656_AD002_indexed_R1_mcc','Pool_841_AD008_indexed_R1_mcc','Pool_587_AD002_indexed_R1_mcc','Pool_735_AD008_indexed_R1_mcc','Pool_1070_AD008_indexed_R1_mcc','Pool_690_AD008_indexed_R1_mcc','Pool_898_AD010_indexed_R1_mcc','Pool_867_AD008_indexed_R1_mcc','Pool_851_AD010_indexed_R1_mcc','Pool_1235_AD010_indexed_R1_mcc','nuclei_540_mcc','Pool_793_AD002_indexed_R1_mcc','Pool_1268_AD008_indexed_R1_mcc','Pool_1160_AD002_indexed_R1_mcc','nuclei_327_mcc','Pool_1116_AD006_indexed_R1_mcc','Pool_888_AD010_indexed_R1_mcc','Pool_747_AD008_indexed_R1_mcc','Pool_688_AD008_indexed_R1_mcc','Pool_757_AD010_indexed_R1_mcc','Pool_753_AD008_indexed_R1_mcc','Pool_731_AD008_indexed_R1_mcc','Pool_1291_AD010_indexed_R1_mcc','Pool_1236_AD008_indexed_R1_mcc','Pool_824_AD006_indexed_R1_mcc','Pool_715_AD010_indexed_R1_mcc','Pool_990_AD002_indexed_R1_mcc','Pool_984_AD006_indexed_R1_mcc','Pool_944_AD002_indexed_R1_mcc','Pool_1266_AD010_indexed_R1_mcc','Pool_1168_AD002_indexed_R1_mcc','Pool_856_AD010_indexed_R1_mcc','Pool_1221_AD008_indexed_R1_mcc','Pool_965_AD006_indexed_R1_mcc','Pool_1195_AD002_indexed_R1_mcc','Pool_1143_AD006_indexed_R1_mcc','Pool_911_AD008_indexed_R1_mcc','Pool_992_AD002_indexed_R1_mcc','Pool_1191_AD002_indexed_R1_mcc','Pool_725_AD008_indexed_R1_mcc','Pool_620_AD002_indexed_R1_mcc','Pool_949_AD002_indexed_R1_mcc','nuclei_514_mcc','Pool_612_AD006_indexed_R1_mcc','Pool_807_AD002_indexed_R1_mcc','Pool_1208_AD010_indexed_R1_mcc','Pool_1158_AD006_indexed_R1_mcc','Pool_723_AD008_indexed_R1_mcc','Pool_1044_AD008_indexed_R1_mcc','Pool_1097_AD010_indexed_R1_mcc','nuclei_599_mcc','Pool_757_AD008_indexed_R1_mcc','Pool_1126_AD002_indexed_R1_mcc','Pool_1132_AD002_indexed_R1_mcc','Pool_1111_AD002_indexed_R1_mcc','Pool_755_AD008_indexed_R1_mcc','Pool_667_AD002_indexed_R1_mcc','Pool_1053_AD010_indexed_R1_mcc','Pool_1010_AD010_indexed_R1_mcc','Pool_1228_AD010_indexed_R1_mcc','Pool_607_AD006_indexed_R1_mcc','Pool_1243_AD010_indexed_R1_mcc','Pool_905_AD008_indexed_R1_mcc','Pool_749_AD010_indexed_R1_mcc','Pool_1224_AD010_indexed_R1_mcc','Pool_1038_AD010_indexed_R1_mcc','Pool_1184_AD002_indexed_R1_mcc','Pool_1238_AD010_indexed_R1_mcc','Pool_1267_AD008_indexed_R1_mcc','Pool_1093_AD010_indexed_R1_mcc','Pool_934_AD002_indexed_R1_mcc','Pool_715_AD008_indexed_R1_mcc','Pool_1293_AD010_indexed_R1_mcc','Pool_982_AD002_indexed_R1_mcc','Pool_808_AD002_indexed_R1_mcc','Pool_1204_AD008_indexed_R1_mcc','Pool_592_AD002_indexed_R1_mcc','Pool_1003_AD006_indexed_R1_mcc','Pool_1289_AD008_indexed_R1_mcc','Pool_973_AD002_indexed_R1_mcc','Pool_818_AD006_indexed_R1_mcc','Pool_626_AD002_indexed_R1_mcc','Pool_624_AD002_indexed_R1_mcc','Pool_627_AD002_indexed_R1_mcc','Pool_717_AD010_indexed_R1_mcc','Pool_1239_AD010_indexed_R1_mcc','Pool_889_AD008_indexed_R1_mcc','nuclei_310_mcc','nuclei_303_mcc','Pool_832_AD006_indexed_R1_mcc','Pool_642_AD006_indexed_R1_mcc','Pool_778_AD006_indexed_R1_mcc','Pool_1204_AD010_indexed_R1_mcc','Pool_851_AD008_indexed_R1_mcc','Pool_585_AD002_indexed_R1_mcc','Pool_994_AD006_indexed_R1_mcc','nuclei_414_mcc','Pool_935_AD006_indexed_R1_mcc','Pool_633_AD006_indexed_R1_mcc','Pool_944_AD006_indexed_R1_mcc','Pool_639_AD006_indexed_R1_mcc','Pool_959_AD006_indexed_R1_mcc','Pool_1255_AD008_indexed_R1_mcc','Pool_1022_AD010_indexed_R1_mcc','Pool_692_AD008_indexed_R1_mcc','Pool_962_AD002_indexed_R1_mcc','Pool_680_AD010_indexed_R1_mcc','Pool_750_AD010_indexed_R1_mcc','Pool_701_AD010_indexed_R1_mcc','Pool_921_AD002_indexed_R1_mcc','Pool_954_AD006_indexed_R1_mcc','Pool_1154_AD002_indexed_R1_mcc','Pool_1073_AD010_indexed_R1_mcc','nuclei_271_mcc','Pool_585_AD006_indexed_R1_mcc','Pool_1232_AD010_indexed_R1_mcc','Pool_826_AD006_indexed_R1_mcc','Pool_977_AD006_indexed_R1_mcc','Pool_686_AD010_indexed_R1_mcc','Pool_668_AD002_indexed_R1_mcc','Pool_1164_AD006_indexed_R1_mcc','Pool_808_AD006_indexed_R1_mcc','Pool_1039_AD008_indexed_R1_mcc','Pool_760_AD008_indexed_R1_mcc','nuclei_527_mcc','Pool_1194_AD006_indexed_R1_mcc','Pool_1123_AD002_indexed_R1_mcc','Pool_1285_AD008_indexed_R1_mcc','Pool_1124_AD002_indexed_R1_mcc','Pool_802_AD006_indexed_R1_mcc','Pool_1084_AD008_indexed_R1_mcc','nuclei_591_mcc','Pool_731_AD010_indexed_R1_mcc','Pool_1166_AD002_indexed_R1_mcc','Pool_1280_AD008_indexed_R1_mcc','Pool_1133_AD006_indexed_R1_mcc','Pool_1198_AD002_indexed_R1_mcc','Pool_1081_AD010_indexed_R1_mcc','Pool_1111_AD006_indexed_R1_mcc','Pool_1067_AD008_indexed_R1_mcc','Pool_850_AD008_indexed_R1_mcc','Pool_718_AD010_indexed_R1_mcc','Pool_845_AD010_indexed_R1_mcc','nuclei_320_mcc','Pool_795_AD002_indexed_R1_mcc','Pool_1007_AD002_indexed_R1_mcc','Pool_1200_AD002_indexed_R1_mcc','Pool_1277_AD010_indexed_R1_mcc','Pool_1127_AD006_indexed_R1_mcc','Pool_1273_AD008_indexed_R1_mcc','Pool_1214_AD010_indexed_R1_mcc','Pool_1237_AD008_indexed_R1_mcc','Pool_980_AD006_indexed_R1_mcc','Pool_898_AD008_indexed_R1_mcc','Pool_692_AD010_indexed_R1_mcc','Pool_739_AD008_indexed_R1_mcc','Pool_788_AD006_indexed_R1_mcc','Pool_597_AD002_indexed_R1_mcc','Pool_1282_AD008_indexed_R1_mcc','Pool_677_AD008_indexed_R1_mcc','Pool_804_AD002_indexed_R1_mcc','Pool_1176_AD002_indexed_R1_mcc','nuclei_305_mcc','Pool_1126_AD006_indexed_R1_mcc','Pool_1091_AD010_indexed_R1_mcc','Pool_983_AD002_indexed_R1_mcc','Pool_1269_AD010_indexed_R1_mcc','Pool_978_AD006_indexed_R1_mcc','Pool_1030_AD010_indexed_R1_mcc','Pool_1129_AD006_indexed_R1_mcc','Pool_656_AD006_indexed_R1_mcc','Pool_1011_AD010_indexed_R1_mcc','Pool_681_AD008_indexed_R1_mcc','Pool_800_AD002_indexed_R1_mcc','Pool_862_AD010_indexed_R1_mcc','Pool_710_AD008_indexed_R1_mcc','Pool_791_AD002_indexed_R1_mcc','Pool_1149_AD002_indexed_R1_mcc','Pool_655_AD002_indexed_R1_mcc','Pool_838_AD006_indexed_R1_mcc','Pool_668_AD006_indexed_R1_mcc','Pool_593_AD006_indexed_R1_mcc','Pool_1220_AD010_indexed_R1_mcc','Pool_1209_AD008_indexed_R1_mcc','Pool_948_AD002_indexed_R1_mcc','Pool_912_AD010_indexed_R1_mcc','Pool_593_AD002_indexed_R1_mcc','Pool_1184_AD006_indexed_R1_mcc','Pool_1061_AD010_indexed_R1_mcc','Pool_1203_AD010_indexed_R1_mcc','Pool_792_AD002_indexed_R1_mcc','Pool_1021_AD008_indexed_R1_mcc','Pool_713_AD008_indexed_R1_mcc','Pool_708_AD008_indexed_R1_mcc','Pool_1163_AD006_indexed_R1_mcc','Pool_601_AD002_indexed_R1_mcc','Pool_665_AD002_indexed_R1_mcc','Pool_828_AD006_indexed_R1_mcc','Pool_636_AD006_indexed_R1_mcc','Pool_887_AD008_indexed_R1_mcc','Pool_934_AD006_indexed_R1_mcc','Pool_842_AD010_indexed_R1_mcc','Pool_594_AD006_indexed_R1_mcc','Pool_813_AD006_indexed_R1_mcc','Pool_836_AD002_indexed_R1_mcc','Pool_1222_AD008_indexed_R1_mcc','Pool_693_AD010_indexed_R1_mcc','Pool_675_AD008_indexed_R1_mcc','Pool_838_AD002_indexed_R1_mcc','Pool_159_AD002_indexed_R1_mcc','Pool_1183_AD008_indexed_R1_mcc','Pool_575_AD010_indexed_R1_mcc','Pool_72_AD002_indexed_R1_mcc','Pool_564_AD008_indexed_R1_mcc','Pool_113_AD006_indexed_R1_mcc','nuclei_380_mcc','Pool_1072_AD008_indexed_R1_mcc','Pool_1159_AD002_indexed_R1_mcc','Pool_440_AD006_indexed_R1_mcc','Pool_869_AD010_indexed_R1_mcc','Pool_1122_AD010_indexed_R1_mcc','nuclei_434_mcc','Pool_1126_AD010_indexed_R1_mcc','Pool_1088_AD010_indexed_R1_mcc','Pool_476_AD002_indexed_R1_mcc','Pool_985_AD002_indexed_R1_mcc','Pool_1186_AD010_indexed_R1_mcc','Pool_461_AD006_indexed_R1_mcc','Pool_385_AD006_indexed_R1_mcc','Pool_970_AD006_indexed_R1_mcc','Pool_768_AD010_indexed_R1_mcc','Pool_842_AD008_indexed_R1_mcc','Pool_260_AD006_indexed_R1_mcc','Pool_1046_AD010_indexed_R1_mcc','nuclei_422_mcc','Pool_1140_AD002_indexed_R1_mcc','Pool_1023_AD008_indexed_R1_mcc','Pool_833_AD002_indexed_R1_mcc','Pool_882_AD008_indexed_R1_mcc','Pool_1237_AD006_indexed_R1_mcc','Pool_548_AD010_indexed_R1_mcc','Pool_373_AD010_indexed_R1_mcc','Pool_1214_AD006_indexed_R1_mcc','Pool_1149_AD010_indexed_R1_mcc','Pool_1228_AD006_indexed_R1_mcc','Pool_1104_AD008_indexed_R1_mcc','Pool_314_AD010_indexed_R1_mcc','Pool_859_AD008_indexed_R1_mcc','Pool_761_AD010_indexed_R1_mcc','Pool_709_AD008_indexed_R1_mcc','Pool_1251_AD010_indexed_R1_mcc','Pool_896_AD010_indexed_R1_mcc','Pool_1059_AD010_indexed_R1_mcc','Pool_157_AD006_indexed_R1_mcc','Pool_1244_AD010_indexed_R1_mcc','Pool_322_AD008_indexed_R1_mcc','Pool_1215_AD010_indexed_R1_mcc','Pool_1279_AD008_indexed_R1_mcc','Pool_1062_AD008_indexed_R1_mcc','Pool_339_AD010_indexed_R1_mcc','Pool_892_AD008_indexed_R1_mcc','Pool_482_AD010_indexed_R1_mcc','Pool_116_AD006_indexed_R1_mcc','Pool_1229_AD008_indexed_R1_mcc','Pool_1229_AD002_indexed_R1_mcc','Pool_132_AD002_indexed_R1_mcc','Pool_1262_AD008_indexed_R1_mcc','SST_A_mcc','Pool_861_AD008_indexed_R1_mcc','Pool_972_AD006_indexed_R1_mcc','Pool_793_AD006_indexed_R1_mcc','Pool_1244_AD002_indexed_R1_mcc','Pool_891_AD008_indexed_R1_mcc','Pool_387_AD006_indexed_R1_mcc','Pool_1237_AD010_indexed_R1_mcc','Pool_1185_AD006_indexed_R1_mcc','Pool_914_AD006_indexed_R1_mcc','Pool_1056_AD008_indexed_R1_mcc','Pool_864_AD008_indexed_R1_mcc','Pool_1177_AD006_indexed_R1_mcc','Pool_50_AD006_indexed_R1_mcc','Pool_1071_AD008_indexed_R1_mcc','Pool_794_AD006_indexed_R1_mcc','Pool_894_AD008_indexed_R1_mcc','Pool_1267_AD010_indexed_R1_mcc','Pool_1253_AD010_indexed_R1_mcc','Pool_758_AD010_indexed_R1_mcc','Pool_1242_AD008_indexed_R1_mcc','Pool_1267_AD002_indexed_R1_mcc','Pool_1258_AD008_indexed_R1_mcc','Pool_1212_AD010_indexed_R1_mcc','Pool_893_AD010_indexed_R1_mcc','Pool_719_AD008_indexed_R1_mcc','Pool_871_AD008_indexed_R1_mcc','Pool_890_AD008_indexed_R1_mcc','Pool_809_AD002_indexed_R1_mcc','nuclei_379_mcc','Pool_1283_AD010_indexed_R1_mcc','Pool_727_AD008_indexed_R1_mcc','Pool_1108_AD006_indexed_R1_mcc','nuclei_417_mcc','Pool_1188_AD006_indexed_R1_mcc','nuclei_431_mcc','Pool_910_AD010_indexed_R1_mcc','Pool_996_AD006_indexed_R1_mcc','Pool_1233_AD008_indexed_R1_mcc','Pool_1165_AD006_indexed_R1_mcc','Pool_904_AD010_indexed_R1_mcc','Pool_1093_AD008_indexed_R1_mcc','Pool_903_AD008_indexed_R1_mcc','Pool_923_AD002_indexed_R1_mcc','Pool_956_AD006_indexed_R1_mcc','nuclei_450_mcc','Pool_891_AD010_indexed_R1_mcc','Pool_1223_AD010_indexed_R1_mcc','Pool_865_AD008_indexed_R1_mcc','AM_P13P14_mcc','AM_P15P16_mcc','Pool_1220_AD006_indexed_R1_mcc','Pool_1060_AD010_indexed_R1_mcc','Pool_576_AD010_indexed_R1_mcc','Pool_478_AD006_indexed_R1_mcc','Pool_502_AD008_indexed_R1_mcc','nuclei_451_mcc','nuclei_433_mcc','Pool_188_AD002_indexed_R1_mcc','Pool_1119_AD008_indexed_R1_mcc','Pool_416_AD002_indexed_R1_mcc','Pool_165_AD002_indexed_R1_mcc','Pool_892_AD010_indexed_R1_mcc','nuclei_446_mcc','nuclei_418_mcc','Pool_833_AD006_indexed_R1_mcc','Pool_517_AD010_indexed_R1_mcc','Pool_899_AD008_indexed_R1_mcc','Pool_846_AD008_indexed_R1_mcc','Pool_1238_AD006_indexed_R1_mcc','Pool_1146_AD010_indexed_R1_mcc','nuclei_454_mcc','Pool_235_AD006_indexed_R1_mcc','Pool_449_AD006_indexed_R1_mcc','Pool_1218_AD006_indexed_R1_mcc','Pool_327_AD010_indexed_R1_mcc','Pool_1248_AD006_indexed_R1_mcc','Pool_150_AD006_indexed_R1_mcc','nuclei_363_mcc','Pool_543_AD010_indexed_R1_mcc','Pool_210_AD002_indexed_R1_mcc','Pool_1242_AD006_indexed_R1_mcc','Pool_1182_AD008_indexed_R1_mcc','Pool_1137_AD008_indexed_R1_mcc','Pool_441_AD006_indexed_R1_mcc','nuclei_423_mcc','Pool_1193_AD010_indexed_R1_mcc','nuclei_439_mcc','Pool_402_AD002_indexed_R1_mcc','Pool_478_AD002_indexed_R1_mcc','Pool_29_AD002_indexed_R1_mcc','Pool_150_AD002_indexed_R1_mcc','Pool_399_AD006_indexed_R1_mcc','Pool_558_AD008_indexed_R1_mcc','Pool_1121_AD008_indexed_R1_mcc','Pool_490_AD010_indexed_R1_mcc','Pool_281_AD002_indexed_R1_mcc','Pool_71_AD002_indexed_R1_mcc','nuclei_442_mcc','Pool_510_AD008_indexed_R1_mcc','Pool_1206_AD006_indexed_R1_mcc','Pool_1273_AD006_indexed_R1_mcc','Pool_839_AD006_indexed_R1_mcc','Pool_334_AD010_indexed_R1_mcc','Pool_435_AD002_indexed_R1_mcc','nuclei_266_mcc','Pool_324_AD010_indexed_R1_mcc','mm_NeuN_neg_male_7wk_mcc','nuclei_409_mcc','Pool_697_AD008_indexed_R1_mcc','Pool_1206_AD008_indexed_R1_mcc','Pool_824_AD002_indexed_R1_mcc','Pool_89_AD002_indexed_R1_mcc','Pool_1058_AD008_indexed_R1_mcc','nuclei_444_mcc','Pool_143_AD006_indexed_R1_mcc','Pool_333_AD010_indexed_R1_mcc','Pool_1057_AD008_indexed_R1_mcc','Pool_1029_AD008_indexed_R1_mcc','Pool_1088_AD008_indexed_R1_mcc','Pool_46_AD006_indexed_R1_mcc','nuclei_452_mcc','Pool_1051_AD008_indexed_R1_mcc','Pool_1106_AD010_indexed_R1_mcc','nuclei_405_mcc','nuclei_390_mcc','Pool_559_AD008_indexed_R1_mcc','Pool_525_AD010_indexed_R1_mcc','Pool_494_AD010_indexed_R1_mcc','Pool_1263_AD002_indexed_R1_mcc','Pool_190_AD002_indexed_R1_mcc','Pool_1278_AD010_indexed_R1_mcc','Pool_1184_AD010_indexed_R1_mcc','Pool_67_AD006_indexed_R1_mcc','Pool_1078_AD010_indexed_R1_mcc','Pool_282_AD002_indexed_R1_mcc','Pool_253_AD006_indexed_R1_mcc','Pool_822_AD006_indexed_R1_mcc','Pool_141_AD006_indexed_R1_mcc','Pool_186_AD002_indexed_R1_mcc','Pool_711_AD010_indexed_R1_mcc','Pool_942_AD006_indexed_R1_mcc','nuclei_419_mcc','Pool_104_AD006_indexed_R1_mcc','Pool_1235_AD006_indexed_R1_mcc','Pool_1293_AD002_indexed_R1_mcc','Pool_77_AD002_indexed_R1_mcc','Pool_345_AD010_indexed_R1_mcc','Pool_479_AD002_indexed_R1_mcc','Pool_1181_AD010_indexed_R1_mcc','Pool_177_AD002_indexed_R1_mcc','Pool_1202_AD002_indexed_R1_mcc','Pool_631_AD002_indexed_R1_mcc','Pool_1147_AD008_indexed_R1_mcc','Pool_453_AD006_indexed_R1_mcc','nuclei_447_mcc','Pool_297_AD010_indexed_R1_mcc','Pool_499_AD010_indexed_R1_mcc','Pool_1240_AD006_indexed_R1_mcc','Pool_1141_AD008_indexed_R1_mcc','Pool_262_AD006_indexed_R1_mcc','Pool_126_AD006_indexed_R1_mcc','Pool_5_AD002_indexed_R1_mcc','Pool_679_AD010_indexed_R1_mcc','Pool_366_AD010_indexed_R1_mcc','Pool_1296_AD002_indexed_R1_mcc','Pool_1294_AD006_indexed_R1_mcc','Pool_34_AD006_indexed_R1_mcc','Pool_1179_AD008_indexed_R1_mcc','Pool_442_AD006_indexed_R1_mcc','nuclei_361_mcc','Pool_303_AD010_indexed_R1_mcc','Pool_378_AD008_indexed_R1_mcc','Pool_348_AD008_indexed_R1_mcc','Pool_264_AD006_indexed_R1_mcc','Pool_88_AD002_indexed_R1_mcc','nuclei_448_mcc','Pool_189_AD006_indexed_R1_mcc','Pool_275_AD006_indexed_R1_mcc','Pool_416_AD006_indexed_R1_mcc','Pool_400_AD006_indexed_R1_mcc','Pool_1260_AD002_indexed_R1_mcc','Pool_158_AD002_indexed_R1_mcc','nuclei_387_mcc','Pool_222_AD006_indexed_R1_mcc','Pool_172_AD002_indexed_R1_mcc','Pool_977_AD002_indexed_R1_mcc','Pool_141_AD002_indexed_R1_mcc','Pool_573_AD008_indexed_R1_mcc','Pool_563_AD008_indexed_R1_mcc','Pool_78_AD006_indexed_R1_mcc','nuclei_430_mcc','Pool_546_AD010_indexed_R1_mcc','Pool_96_AD002_indexed_R1_mcc','Pool_1134_AD010_indexed_R1_mcc','Pool_511_AD008_indexed_R1_mcc','Pool_1171_AD010_indexed_R1_mcc','Pool_162_AD006_indexed_R1_mcc','Pool_467_AD006_indexed_R1_mcc','Pool_532_AD010_indexed_R1_mcc','Pool_1099_AD010_indexed_R1_mcc','nuclei_432_mcc','Pool_848_AD008_indexed_R1_mcc','Pool_439_AD006_indexed_R1_mcc','Pool_872_AD010_indexed_R1_mcc','Pool_561_AD010_indexed_R1_mcc','Pool_1079_AD010_indexed_R1_mcc','nuclei_449_mcc','Pool_547_AD010_indexed_R1_mcc','Pool_1260_AD010_indexed_R1_mcc','Pool_483_AD010_indexed_R1_mcc','Pool_1128_AD010_indexed_R1_mcc','Pool_505_AD008_indexed_R1_mcc','Pool_496_AD008_indexed_R1_mcc','Pool_315_AD010_indexed_R1_mcc','Pool_1123_AD008_indexed_R1_mcc','Pool_162_AD002_indexed_R1_mcc','Pool_343_AD010_indexed_R1_mcc','Pool_869_AD008_indexed_R1_mcc','Pool_437_AD006_indexed_R1_mcc','Pool_452_AD006_indexed_R1_mcc','Pool_845_AD008_indexed_R1_mcc','nuclei_406_mcc','Pool_101_AD006_indexed_R1_mcc','Pool_99_AD002_indexed_R1_mcc','Pool_1128_AD008_indexed_R1_mcc','nuclei_456_mcc','Pool_1074_AD010_indexed_R1_mcc','Pool_794_AD002_indexed_R1_mcc','Pool_1268_AD006_indexed_R1_mcc','Pool_461_AD002_indexed_R1_mcc','Pool_1222_AD006_indexed_R1_mcc','Pool_10_AD002_indexed_R1_mcc','Pool_1241_AD002_indexed_R1_mcc','Pool_86_AD006_indexed_R1_mcc','Pool_542_AD008_indexed_R1_mcc','Pool_1240_AD008_indexed_R1_mcc','AM_V9_mcc','AM_V8_mcc','nuclei_426_mcc','Pool_128_AD002_indexed_R1_mcc','Pool_1195_AD008_indexed_R1_mcc','Pool_1120_AD008_indexed_R1_mcc','Pool_391_AD002_indexed_R1_mcc','Pool_166_AD006_indexed_R1_mcc','Pool_1264_AD002_indexed_R1_mcc','Pool_1295_AD006_indexed_R1_mcc','nuclei_436_mcc','Pool_1201_AD006_indexed_R1_mcc','Pool_281_AD006_indexed_R1_mcc','Pool_1005_AD002_indexed_R1_mcc','Pool_402_AD006_indexed_R1_mcc','Pool_573_AD010_indexed_R1_mcc','Pool_1160_AD010_indexed_R1_mcc','Pool_454_AD006_indexed_R1_mcc','Pool_22_AD002_indexed_R1_mcc','Pool_458_AD006_indexed_R1_mcc','Pool_8_AD002_indexed_R1_mcc','Pool_554_AD008_indexed_R1_mcc','nuclei_435_mcc','Pool_1109_AD010_indexed_R1_mcc','Pool_1115_AD008_indexed_R1_mcc','Pool_282_AD006_indexed_R1_mcc','Pool_544_AD008_indexed_R1_mcc','Pool_360_AD010_indexed_R1_mcc','Pool_1247_AD002_indexed_R1_mcc','Pool_247_AD006_indexed_R1_mcc','Pool_226_AD002_indexed_R1_mcc','Pool_1117_AD006_indexed_R1_mcc','Pool_138_AD006_indexed_R1_mcc','Pool_471_AD006_indexed_R1_mcc','Pool_114_AD006_indexed_R1_mcc','Pool_22_AD006_indexed_R1_mcc','Pool_94_AD006_indexed_R1_mcc','Pool_405_AD006_indexed_R1_mcc','Pool_93_AD006_indexed_R1_mcc','Pool_1124_AD008_indexed_R1_mcc','Pool_406_AD006_indexed_R1_mcc','Pool_515_AD008_indexed_R1_mcc','Pool_1133_AD010_indexed_R1_mcc','Pool_712_AD008_indexed_R1_mcc','Pool_493_AD008_indexed_R1_mcc','Pool_1168_AD010_indexed_R1_mcc','Pool_1227_AD010_indexed_R1_mcc','Pool_85_AD006_indexed_R1_mcc','Pool_1258_AD002_indexed_R1_mcc','Pool_78_AD002_indexed_R1_mcc','Pool_407_AD002_indexed_R1_mcc','Pool_1244_AD006_indexed_R1_mcc','Pool_1137_AD006_indexed_R1_mcc','Pool_1193_AD006_indexed_R1_mcc','nuclei_280_mcc','Pool_301_AD010_indexed_R1_mcc','Pool_530_AD008_indexed_R1_mcc','Pool_1172_AD010_indexed_R1_mcc','Pool_274_AD006_indexed_R1_mcc','Pool_693_AD008_indexed_R1_mcc','Pool_781_AD006_indexed_R1_mcc','Pool_1179_AD010_indexed_R1_mcc','Pool_914_AD002_indexed_R1_mcc']])

def spinit(df, num_genes, var_increase, min_cluster_size):

    df = df.loc[df.var(axis=1).sort_values(ascending=False).index.tolist()[:num_genes]]
    df.reset_index(inplace=True, drop=True)
    # df = df.subtract(df.mean(axis=1), axis=0)
    # ipdb.set_trace()

    data = df.subtract(df.mean(axis=1), axis=0).values

    # data = df.values
    # data = data[np.argsort(data.var(1))[::-1][:num_genes],:]

    # data = data - data.mean(1)[:,newaxis]

    results = SPIN(data, widlist=runs_step, iters=runs_iters, axis=normal_spin_axis, verbose=verbose)
    # results = SPIN(df.values, widlist=runs_step, iters=runs_iters, axis=normal_spin_axis, verbose=False)

    cells = [df.columns.tolist()[ind] for ind in results[1]]

    df = df[cells]
    scores = np.zeros(df.shape[1])-1
    corrmat = df.corr().values

    print("Finding cut point...")
    for i in range(2, df.shape[1]-2):
        scores[i] = (sum(corrmat[:i,:i]) + sum(corrmat[i:,i:])) / float(i**2 + (df.shape[1]-i)**2)

    split_point = np.argmax(scores)
    print("Cut point found: " + str(split_point))

    clust1 = cells[:split_point]
    clust2 = cells[split_point:]
    # print("Size clust1: " + str(clust1))
    # print("Size clust2: " + str(clust2))

    # # ipdb.set_trace()
    # num_sig = 0.0
    # print('\nChecking for significant genes...:')
    # for i in range(20):
    #     # _, pvalue = stats.ttest_ind(data[i,split_point:], data[i,:split_point])
    #     _, pvalue = stats.ttest_ind(df.loc[i,clust1], df.loc[i,clust2])
    #     print(pvalue)
    #     # pval = stats.ttest_ind(data[i,split_point:],data[i,:split_point])
    #     if  pvalue < .05:
    #         num_sig += 1
    # perc_sig = num_sig/20
    # print('Percent sig: ' + str(perc_sig))

    # ipdb.set_trace()

    # plt.imshow(corrmat)
    # plt.axvline(split_point, color='k', linestyle='--')
    # plt.axhline(split_point, color='k', linestyle='--')
    # plt.colorbar()
    # plt.show()

    inc_clust1 = corrmat[:split_point,:split_point].mean()/corrmat.mean()
    inc_clust2 = corrmat[split_point:,split_point:].mean()/corrmat.mean()

    print('\nIncrease clust 1: '+str(inc_clust1))
    print('Increase clust 2: '+str(inc_clust2))
    print('Max: '+str(max(inc_clust1, inc_clust2)))

    # if perc_sig > .4:
    if max(inc_clust1, inc_clust2) > var_increase:
        return clust1, clust2
    else:
        return cells, []

    # if max(inc_clust1, inc_clust2) > var_increase:
    # if num_sig
    # if (corrmat[:split_point,:split_point].shape[0] < min_cluster_size) or (corrmat[split_point:,split_point:].shape[0] < min_cluster_size):
    #         return cells, []
    #     else:
    #         return clust1, clust2
    # else:
    #     return cells, []


num_genes = 2000
min_cluster_size = 50
var_increase = 1.15

keep_splitting = True
level = 0 # EAM

while keep_splitting:
    print("\n\n\nProcessing level " + str(level) + "...")
    keep_splitting = False
    prev_level = all_levels[level]
    curr_level = []

    for cluster in prev_level:
        if len(cluster) <= min_cluster_size:
            print("Cluster is too small")
            curr_level.append(cluster)
        else:
            df_clust = df[cluster]
            clust1, clust2 = spinit(df_clust, num_genes, var_increase, min_cluster_size)
            if len(clust2) == 0:
                print("Cluster failed to split. Too few sig genes.")
                curr_level.append(clust1)
            else:
                print("Cluster split.")
                keep_splitting = True
                curr_level.append(clust1)
                curr_level.append(clust2)

    all_levels.append(curr_level)
    level += 1


with open(filename+'_backSPIN_clusters.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(all_levels)


# # with open('res_sig_test.csv', 'w') as f:
# #     writer = csv.writer(f)
# #     writer.writerows(all_levels)

# # # num_genes = 1000
# # # # with open('res_'+str(num_genes)+'.csv', 'r') as f:
# # # # with open('res_sig_test.csv', 'r') as f:
# # # with open("res_"+str(num_genes)+"_thresh1_min50.csv", "w") as f:
# # #     reader = csv.reader(f)
# # #     all_levels = list(reader)

# # # num_genes = 500

# # with open('res_clustsize200.csv', 'r') as f:

# with open('res_siggenes.6.csv', 'r') as f :
# with open('res_2000top_1.15var.csv', 'r') as f :
# with open('res_2000top_.4siggenes.csv', 'r') as f :
with open(filename+'_backSPIN_clusters.csv', 'r') as f :
    reader = csv.reader(f)
    all_levels = list(reader)

#
#
# df_TSNE = pd.read_csv('/cndd/ckeown/single_cells/analysis/dashboard_clustering_perp40_numbinsall_all_PCA50_tSNE2_bins_normalized_notcorrected_filtered_eps1.2_mouse.txt', sep="\t")
# df_TSNE.samp = df_TSNE.samp + '_mcc'
# df_TSNE.tsne_y = df_TSNE.tsne_y * -1
#
# fig, ax = plt.subplots(3,3, figsize=(10,10))
# for i in range(0,len(all_levels)):
#     clusters = all_levels[i]
#     cols = mypy.get_random_colors(len(clusters))
#     for j,cluster in enumerate(clusters):
#         df_tmp = df_TSNE.loc[df_TSNE.samp.isin(cluster.split("', '"))]
#         ax[np.unravel_index(i, (3,3))].scatter(df_tmp.tsne_x, df_tmp.tsne_y, c=cols[j], marker='.', edgecolor='none')
#     ax[np.unravel_index(i, (3,3))].set_xticks([])
#     ax[np.unravel_index(i, (3,3))].set_yticks([])
#     ax[np.unravel_index(i, (3,3))].set_title('Level '+str(i))
# for j in range(i+1,9):
#     ax[np.unravel_index(j, (3,3))].axis('off')
# # fig.suptitle('Split by requiring >40% of top-most variable genes be sig diff methylated')
# fig.suptitle('Split by requiring >15% increase in average correlation within one of the child clusters over the parent')
# # plt.tight_layout()
# # plt.show()
# # plt.savefig('fig_mouse_40perc_sigdiffmeth.pdf', format='pdf')
# plt.savefig('fig_mouse_15perc_inc_corr.pdf', format='pdf')
# # plt.savefig('backSPIN_results_'+str(num_genes)+'genes.pdf')
#



# # # fig, ax = plt.subplots(1,1)
# # # i = 8
# # # clusters = all_levels[i]
# # # cols = mypy.get_random_colors(len(clusters))
# # # for j,cluster in enumerate(clusters):
# # #     df_tmp = df_TSNE.loc[df_TSNE.samp.isin(cluster.split("', '"))]
# # #     ax.scatter(df_tmp.tsne_x, df_tmp.tsne_y, c=cols[j], marker='.', edgecolor='none')
# # # ax.set_xticks([])
# # # ax.set_yticks([])
# # # ax.set_title('Level '+str(i))
# # # plt.show()
