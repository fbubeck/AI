?&$	??&??????"????[?kBZc??!?8毐??$	QbRe?
@a?ެ?????~??@!F|&ֿk@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0J&?v???#N'??r??A??e?c]??Y????ɓ?rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0,??yp???$?@????A?
??????Y?Z?Qf??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????
???EIH?m???A?x@ٔ??Y؀q????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails037߈??????yȔ??A#M?<i??Y??Xİ??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails06??`?
??%=?N???A?????Y,?????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0(??vL???)? ?h??ANB?!???Y???UG???rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?1?Mc????O ????A?}r 
??Y???4??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?????????)1	??A?.?.???Y???8Q??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
scz????T?:???A?,??\n??YA}˜.???rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?8毐??0???"??A??7h???YR*?	????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?M?????'?;??A?-??e???Yvl?u???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?h?????uۈ'???AZ??????Y֪]???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0n??Ũk???z????A??捓???Y???0????rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0[?kBZc???o'?_??A??&k?C??Yط???/??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails02??Y?S???T3k) ??A??;F??YqY?? ??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???X????w?'-??A5????K??Y:???u??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???S???????N@??AǺ?????Y??#?????rtrain 97*	/?$W?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatF?Swe??!I?c]?@@)?~?Ϛ??1f?`(?>@:Preprocessing2E
Iterator::RootT???=??!?]4??!B@)???p?Q??1?+ײ?3@:Preprocessing2T
Iterator::Root::ParallelMapV2 C?*??!?J?ҿ0@) C?*??1?J?ҿ0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice'?%??s??!??$n,?"@)'?%??s??1??$n,?"@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?<0????!'??F=?O@)?t?? ???1;Ω?""@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate;??]ض?!??g?Si.@)P???<??1Wd?DN?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??ǁW??!?}?n?4@)X??G???1f(??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*y??.??!???9?4@)y??.??1???9?4@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 43.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9X??w?@IeA??'X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	j< OL??8;??~????o'?_??!0???"??	!       "	!       *	!       2$	?ޱ?B??Y?@ه?????&k?C??!??;F??:	!       B	!       J$	?l0????v??\q?ط???/??!???4??R	!       Z$	?l0????v??\q?ط???/??!???4??b	!       JCPU_ONLYYX??w?@b qeA??'X@Y      Y@q?G??DN@"?	
both?Your program is POTENTIALLY input-bound because 43.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?60.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 