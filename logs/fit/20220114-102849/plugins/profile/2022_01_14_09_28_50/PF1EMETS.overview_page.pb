?&$	z???:???uyb*???K6l???!???.???$	s?K}/?@]"?s????B????@!? r F@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?d?F ?????N??D??AÝ#????Y?IEc????rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????=??@fg??A???q?j??Y?ّ?;???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?=??j??????:8??A?j?????Y*oG8-x??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0$}ZE???YL???A%xC8??Y#??E????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0>!;oc3??y?t?????AC??6??Y<i??
??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??1zn???v????A;?I/???YX???<???rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?*?w?????2T?T???A??)????Y_?"??]??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?s????[rP?L??A?;ۣ7???Y?-??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?K6l?????H?[??A@?P?%??Y6<?R?!??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Y?wg?????/fK??A?@???F??Yۊ?e????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails04-?2???.??e?O??A???????Y??u?T??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???@-????5?????A??)????Y`?_????rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???
??0J?_???A?zj??U??YS?'?ݚ?rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???.???Ac&Q/???A?"h?$???Y?pu ?]??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0M?]~??Y???RA??A?]=???Y??2R臭?rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0+?C3O.?????????A??????Y?:?f???rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???l!??uXᖏ???A??i????Y4?c?=	??rtrain 97*	?G?z??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatp?>;????!a????KE@)cG?P???1\?r?|?C@:Preprocessing2T
Iterator::Root::ParallelMapV2?!??3???!???U/@)?!??3???1???U/@:Preprocessing2E
Iterator::Root??8*7Q??!?h???>@)?O?}:??1??k?v.@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?$\?#???!ۥ[\?MQ@)x?=\rܵ?1P???j!@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?(?7ӵ?!??y8pc!@)?(?7ӵ?1??y8pc!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate]?z???!????,?*@)["??ߧ?1?i?x@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?f?|?|??!??~?1@)?'d?ml??1+Cp!??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?0?????!P????w@)?0?????1P????w@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9#O???K@I??"9?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	мx??s????f??????H?[??!y?t?????	!       "	!       *	!       2$	&4?d????P_K??@?P?%??!?"h?$???:	!       B	!       J$	? |?!??????k#i?S?'?ݚ?!??u?T??R	!       Z$	? |?!??????k#i?S?'?ݚ?!??u?T??b	!       JCPU_ONLYY#O???K@b q??"9?X@Y      Y@q???5.aG@"?	
both?Your program is POTENTIALLY input-bound because 51.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?46.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 