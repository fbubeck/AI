?&$	?b?Q }??????8??R+L?k??!?Ѫ?t???$	?̐?@d@?r?????+^8	@!0??k??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails00??!????o{????A0?[w?T??Y3???yS??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0bۢ????)?'?$???A????)??Y?M?????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0yܝ????dY0?GQ??Ac??????Y=֌r??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0D?H?????a\:?<??Ap?^}<???Y???F???rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0D? ???+?m?????A"???k???Y?F??Ǒ?rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?9Dܜ??????&???AJ?E???Y???????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0%?}?e????M?G????A?HJzZ??Yp^??jG??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	R+L?k????h?x???A??.?.??Y?V?9?m??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
??S9?)?????G???A??1=a???Y???????rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?q75??.?R\U???AdT8?T??YJ???????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0.?5#??2q? ???Ar?@H0??Y?+?j???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Ѫ?t????????b??A??\??k??Y??a????rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?u7Ou?????}?p??A:?V?S??Y?Sr3ܐ?rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?8)?{????ٮ????A?YL??Yn??t???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0M???D?????ȭI??AwKr??&??Ys??c?Ȑ?rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???"R??HߤiP4??A?KK??Y$?6?De??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Gߤi??@?5_%??A??	h"l??Y? ?bG???rtrain 97*	@5^?I??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatYk(????!=?!zpcC@)r 
fL??1E???=?@@:Preprocessing2E
Iterator::Root?۞ ????!?3A?9??@)?6 !??1Vx?Ȥ?/@:Preprocessing2T
Iterator::Root::ParallelMapV2k?=&R???!????</@)k?=&R???1????</@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipc??*3???!???qQ@)?sE)!X??1z???>7%@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????!????!@)????1????!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate	????Q??!?t?"%?.@)_Pj??1??u'?$@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*5?BX?%??!??q)?Y@)5?BX?%??1??q)?Y@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapMK??F??!???E?2@)?[z4Փ?1?eᚭ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?7??wV@IC??ALX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?V???p??)?ضO????M?G????!@?5_%??	!       "	!       *	!       2$	??F??`????	?????dT8?T??!??\??k??:	!       B	!       J$	n????G????K?-?E???a????!$?6?De??R	!       Z$	n????G????K?-?E???a????!$?6?De??b	!       JCPU_ONLYY?7??wV@b qC??ALX@Y      Y@q'?T?I3U@"?	
both?Your program is POTENTIALLY input-bound because 49.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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