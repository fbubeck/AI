?&$	(?? 9????*????ڦx\T??!N?t"???$	???*Z?@*O??????ͬc]@!?`?ot@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0/kb??????cw????A?y????Y~??g??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0N?t"????p;4,F??A'L?????YM??~?T??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?p!????je?/??A8?*5{???Y??iT???rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0؛?????k?3???A?	0,???Y??W??͔?rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0a???U???Y????A?]ؚ????Y@ޫV&???rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??'??????A?V?9??A???J????Y??8h??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?M?=????]?z???A?]??k??Y???;3??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	????M??q???im??Aj1x?????YN???????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
.7?????W?"????A??nI???Yc}?E??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Z??U?P???d73????A?fb????Y?T?????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?X6sHj???fG?????A?($??;??Y?LM?7???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Y??9??Eh??5??An??ʆ5??Y??M(D??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0_A??h:??£?#???A???AB???Y??f?\S??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?mm?y?????m3???A?????@??Y???~??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ڦx\T???@?M?G??A :̗`??Y??_ ??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0-\Va3????W}??AZ??/-???Y#???Sɐ?rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?	?i?????I?_{??A"ĕ?wF??Y,,?????rtrain 97*	'1?J?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?<?$??!?]?9uB@)??E??(??1??w??A@:Preprocessing2E
Iterator::Rootū?m????!P?-??@@)r4GV~??1@?n?1@:Preprocessing2T
Iterator::Root::ParallelMapV2#??u??!^I<+?0@)#??u??1^I<+?0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?%TpxA??!???#@)?%TpxA??1???#@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip6!?1????!Xi??P@)X?\T??1?u?+?#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate????岵?!m?Y???.@)?c???H??1?(?L?a@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?捓¼??!Lݴf?3@)??O?s'??1T??|?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?Ue????!??Y?ok@)?Ue????1??Y?ok@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?J?~@I?u??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	(?9)#???X?ꏠ???m3???!?p;4,F??	!       "	!       *	!       2$	?G???\??????????J????!'L?????:	!       B	!       J$	???h#????l
(Qp???_ ??!?T?????R	!       Z$	???h#????l
(Qp???_ ??!?T?????b	!       JCPU_ONLYY?J?~@b q?u??X@Y      Y@q??ZQj5U@"?	
both?Your program is POTENTIALLY input-bound because 48.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?84.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 