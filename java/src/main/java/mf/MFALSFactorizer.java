/* 
 * Copyright (C) 2015 Information Retrieval Group at Universidad Autónoma
 * de Madrid, http://ir.ii.uam.es
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
package mf;

import cern.colt.function.DoubleFunction;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.mf.Factorization;
import es.uam.eps.ir.ranksys.mf.Factorizer;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;

import static java.lang.Math.sqrt;

/**
 * Notice that this class has been developed by Saul Vargas under RankSys package, we only modified it by ading a random seed
 * such that the experiments can be reproducible! - Mesut Kaya
 *
 * Generic alternating least-squares factorizer.
 *
 * @author Saúl Vargas (saul.vargas@uam.es)
 *
 * @param <U> type of the users
 * @param <I> type of the items
 */
public abstract class MFALSFactorizer<U, I> extends Factorizer<U, I> {

    private static final Logger LOG = Logger.getLogger(es.uam.eps.ir.ranksys.mf.als.ALSFactorizer.class.getName());

    private final int numIter;

    /**
     * Constructor.
     *
     * @param numIter number of least-squares calculations
     */
    public MFALSFactorizer(int numIter) {
        this.numIter = numIter;
    }

    @Override
    public double error(Factorization<U, I> factorization, FastPreferenceData<U, I> data) {

        DenseDoubleMatrix2D p = factorization.getUserMatrix();
        DenseDoubleMatrix2D q = factorization.getItemMatrix();

        return error(p, q, data);
    }

    @Override
    public Factorization<U, I> factorize(int K, FastPreferenceData<U, I> data) {
        long seed = 1987;
        Random random = new Random(seed);
        random.nextDouble();
        //DoubleFunction init = x -> sqrt(1.0 / K) * Math.random();
        DoubleFunction init = x -> sqrt(1.0 / K) * random.nextDouble();
        Factorization<U, I> factorization = new Factorization<>(data, data, K, init);
        factorize(factorization, data);
        return factorization;
    }

    @Override
    public void factorize(Factorization<U, I> factorization, FastPreferenceData<U, I> data) {

        DenseDoubleMatrix2D p = factorization.getUserMatrix();
        DenseDoubleMatrix2D q = factorization.getItemMatrix();

        IntSet uidxs = new IntOpenHashSet(data.getUidxWithPreferences().toArray());
        IntStream.range(0, p.rows()).filter(uidx -> !uidxs.contains(uidx)).forEach(uidx -> p.viewRow(uidx).assign(0.0));
        IntSet iidxs = new IntOpenHashSet(data.getIidxWithPreferences().toArray());
        IntStream.range(0, q.rows()).filter(iidx -> !iidxs.contains(iidx)).forEach(iidx -> q.viewRow(iidx).assign(0.0));

        for (int t = 1; t <= numIter; t++) {
            long time0 = System.nanoTime();

            set_minQ(q, p, data);
            set_minP(p, q, data);

            int iter = t;
            long time1 = System.nanoTime() - time0;

            LOG.log(Level.INFO, String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.log(Level.FINE, () -> String.format("iteration n = %3d e = %.6f", iter, error(factorization, data)));
        }
    }

    /**
     * Squared loss of two matrices.
     *
     * @param p user matrix
     * @param q item matrix
     * @param data preference data
     * @return squared loss
     */
    protected abstract double error(DenseDoubleMatrix2D p, DenseDoubleMatrix2D q, FastPreferenceData<U, I> data);

    /**
     * User matrix least-squares step.
     *
     * @param p user matrix
     * @param q item matrix
     * @param data preference data
     */
    protected abstract void set_minP(DenseDoubleMatrix2D p, DenseDoubleMatrix2D q, FastPreferenceData<U, I> data);

    /**
     * Item matrix least-squares step.
     *
     * @param q item matrix
     * @param p user matrix
     * @param data preference data
     */
    protected abstract void set_minQ(DenseDoubleMatrix2D q, DenseDoubleMatrix2D p, FastPreferenceData<U, I> data);
}
