?&$	??t????H0?????˅ʿ?W??!????????$	<?c?W@?n?2Eu??*[????@!?^??d@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?K8???????n,(??An½2o???Y#?k$	??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?JU???i???n??A??ݓ????Y?u??Xߐ?rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ZH?????H?C?????A?S??Yh??Y??tB???rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0˅ʿ?W??????bc??AB"m?OT??Y?9d?w??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0؂?C ????}U.T??A׾?^?s??YŮ??????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?h[?:??_@/ܹ0??A/?????Y?????R??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0O??e??U/??d???A'N?w(
??Y??y7??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?j+??]??X??V????A+??O8???Y?a??c??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
	À%W???????}r??A???Ĭ??Yh	2*??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????????G???A*?Z^???YԛQ?U???rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0p?܁???G>?x????AC?*q??Y??O????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????q75??AYl??????Y???????rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?D.8??????G??A2*A*??Y???n??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?O ????#/kb????AW?sD?K??YQi??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0%?}?e????cw????A;??bFx??Y????#??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails02?g?o???<?͌~4??A`??5!???Y|?i?????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0H?}8g??&?fe???AM2r????Y?1??????rtrain 97*	Zd;߁?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???K??!??ʗ?B@)n???+??1???->e@@:Preprocessing2E
Iterator::Root?M?a????!?????;@@)p|??%??16??C?E0@:Preprocessing2T
Iterator::Root::ParallelMapV2?)????!/?e?/20@)?)????1/?e?/20@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip^??I????!'??;	?P@)h???bE??1~??@M?%@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceI?p??!P????H#@)I?p??1P????H#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??ihw??!찌 E.@)??x????1?LdE?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*e???\???!?O???z@)e???\???1?O???z@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??/????!?oQ?NL3@)?2#???1#???9?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9g(
Q??
@I??w)X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	h?O???O????&?fe???!i???n??	!       "	!       *	!       2$	/-??#??"?L????/?????!*?Z^???:	!       B	!       J$	?S"7q ???G??sp??a??c??!Ů??????R	!       Z$	?S"7q ???G??sp??a??c??!Ů??????b	!       JCPU_ONLYYg(
Q??
@b q??w)X@Y      Y@q??l
~?T@"?	
both?Your program is POTENTIALLY input-bound because 46.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?83.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 